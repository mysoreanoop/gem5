/*
 * Copyright (c) 2014-2015 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __LDS_STATE_HH__
#define __LDS_STATE_HH__

#include <array>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "debug/GPULDS.hh"
#include "gpu-compute/misc.hh"
#include "mem/port.hh"
#include "params/LdsState.hh"
#include "sim/clocked_object.hh"

namespace gem5
{

class ComputeUnit;

/**
 * this represents a slice of the overall LDS, intended to be associated with
 * an individual workgroup
 */
class LdsChunk
{
  public:
    LdsChunk(uint32_t x_size, int dispatch_id, int wg_id, int wf_cnt,
                uint32_t reserve=0):
        chunk(x_size),
        _upgraded(false),
        _terminal(false),
        _downgradePosts(0),
        _terminalPosts(0),
        dispatchId(dispatch_id),
        wgId(wg_id),
        numWfs(wf_cnt),
        reservedSpace(reserve)
    {
      DPRINTF(GPULDS, "Initializing chunk with size %d, reserved %d\n",
        x_size, reservedSpace);
    }

    LdsChunk() {}

    int getReservedSpace() {
      return reservedSpace;
    }

    bool isTerminal() {
      return _terminal;
    }

    bool upgrade() {
      if (!_upgraded) {
        DPRINTF(GPULDS, "LDS[%d][%d] upgrade to %d\n",
                  dispatchId, wgId,
                  chunk.size() + reservedSpace);
        chunk.resize(chunk.size() + reservedSpace);
        _upgraded = true;
        reservedSpace = 0; // reset
        return true;
      } else {
        // is this even possible?
        DPRINTF(GPULDS, "LDS[%d][%d] already upgraded\n",
                  dispatchId, wgId);
        return false;
      }
    }

    bool downgrade(const uint32_t to) {
      assert (_downgradePosts < numWfs);
      if (++_downgradePosts == numWfs) {
        DPRINTF(GPULDS, "LDS[%d][%d] downgrade to %d\n",
                  dispatchId, wgId, to);
        chunk.resize(to);
        _downgradePosts = 0;
        return true;
      } else {
        DPRINTF(GPULDS, "LDS[%d][%d] %d WFs remain\n",
                dispatchId, wgId,
                numWfs - _downgradePosts);
        return false;
      }
    }

    bool markTerminal() {
      assert (_terminalPosts < numWfs);
      if (++_terminalPosts == numWfs) {
        DPRINTF(GPULDS, "LDS[%d][%d] marked terminal\n",
                  dispatchId, wgId);
        _terminal = true;
        _terminalPosts = 0;
        return true;
      } else {
        DPRINTF(GPULDS, "LDS[%d][%d] %d WFs remain\n",
                dispatchId, wgId,
                numWfs - _terminalPosts);
        return false;
      }

    }
    /**
     * a read operation
     */
    template<class T>
    T
    read(const uint32_t index)
    {
        /**
         * For reads that are outside the bounds of the LDS
         * chunk allocated to this WG we return 0.
         */
        if (index >= chunk.size()) {
            DPRINTF(GPULDS, "LDS[%d][%d]: Read %d beyond size (%ld)\n",
                    dispatchId, wgId, index, chunk.size());
            return (T)0;
        }

        T *p0 = (T *) (&(chunk.at(index)));

        if (sizeof(T) <= 4) {
            [[maybe_unused]] uint32_t int_val =
                *reinterpret_cast<uint32_t*>(p0);
            DPRINTF(GPULDS, "LDS[%d][%d]: Read %08x from index %d\n",
                    dispatchId, wgId, int_val, index);
        } else if (sizeof(T) <= 8) {
            [[maybe_unused]] uint64_t int_val =
                *reinterpret_cast<uint64_t*>(p0);
            DPRINTF(GPULDS, "LDS[%d][%d]: Read %016lx from index %d\n",
                    dispatchId, wgId, int_val, index);
        } else if (sizeof(T) <= 16) {
            [[maybe_unused]] uint64_t *int_vals =
                reinterpret_cast<uint64_t*>(p0);
            DPRINTF(GPULDS, "LDS[%d][%d]: Read %016lx%016lx from index %d\n",
                    dispatchId, wgId, int_vals[1], int_vals[0], index);
        }

        return *p0;
    }

    /**
     * a write operation
     */
    template<class T>
    void
    write(const uint32_t index, const T value)
    {
        /**
         * Writes that are outside the bounds of the LDS
         * chunk allocated to this WG are dropped.
         */
        if (index >= chunk.size()) {
            DPRINTF(GPULDS, "LDS[%d][%d]: Ignoring write beyond current size "
                    "(%ld)\n", dispatchId, wgId, chunk.size());
            // fatal("save me");
            return;
        }

        T *p0 = (T *) (&(chunk.at(index)));

        if (sizeof(T) <= 4) {
            [[maybe_unused]] uint32_t prev_val =
                *reinterpret_cast<uint32_t*>(p0);
            DPRINTF(GPULDS, "LDS[%d][%d]: Write %08lx to index %d (was "
                   "%08lx)\n", dispatchId, wgId, value, index, prev_val);
        } else if (sizeof(T) <= 8) {
            [[maybe_unused]] uint64_t prev_val =
                *reinterpret_cast<uint64_t*>(p0);
            DPRINTF(GPULDS, "LDS[%d][%d]: Write %016lx to index %d (was "
                   "%016lx)\n", dispatchId, wgId, value, index, prev_val);
        } else if (sizeof(T) <= 16) {
            [[maybe_unused]] uint64_t *prev_vals =
                reinterpret_cast<uint64_t*>(p0);
            [[maybe_unused]] const uint64_t *next_vals =
                reinterpret_cast<const uint64_t*>(&value);
            DPRINTF(GPULDS, "LDS[%d][%d]: Write %016lx%016lx to index %d "
                    "(was %016lx%016lx)\n", dispatchId, wgId, next_vals[1],
                    next_vals[0], index, prev_vals[1], prev_vals[0]);
        }

        *p0 = value;
    }

    /**
     * an atomic operation
     */
    template<class T>
    T
    atomic(const uint32_t index, AtomicOpFunctorPtr amoOp)
    {
        /**
         * Atomics that are outside the bounds of the LDS
         * chunk allocated to this WG are dropped.
         */
        if (index >= chunk.size()) {
            return (T)0;
        }
        T *p0 = (T *) (&(chunk.at(index)));
        T tmp = *p0;

       (*amoOp)((uint8_t *)p0);
        return tmp;
    }

    /**
     * get the size of this chunk
     */
    std::vector<uint8_t>::size_type
    size() const
    {
        return chunk.size();
    }


  protected:
    // the actual data store for this slice of the LDS
    std::vector<uint8_t> chunk;
    bool _upgraded;
    bool _terminal;


  private:
    int _downgradePosts;
    int _terminalPosts;

    uint32_t dispatchId;
    uint32_t wgId;
    int numWfs;
    int reservedSpace;
};

// Local Data Share (LDS) State per Wavefront (contents of the LDS region
// allocated to the WorkGroup of this Wavefront)
class LdsState: public ClockedObject
{
  protected:

    /**
     * an event to allow event-driven execution
     */
    class TickEvent: public Event
    {
      protected:

        LdsState *ldsState = nullptr;

        Tick nextTick = 0;

      public:

        TickEvent(LdsState *_ldsState) :
            ldsState(_ldsState)
        {
        }

        virtual void
        process();

        void
        schedule(Tick when)
        {
            mainEventQueue[0]->schedule(this, when);
        }

        void
        deschedule()
        {
            mainEventQueue[0]->deschedule(this);
        }
    };

    /**
     * CuSidePort is the LDS Port closer to the CU side
     */
    class CuSidePort: public ResponsePort
    {
      public:
        CuSidePort(const std::string &_name, LdsState *_ownerLds) :
                ResponsePort(_name), ownerLds(_ownerLds)
        {
        }

      protected:
        LdsState *ownerLds;

        virtual bool
        recvTimingReq(PacketPtr pkt);

        virtual Tick
        recvAtomic(PacketPtr pkt)
        {
          return 0;
        }

        virtual void
        recvFunctional(PacketPtr pkt);

        virtual void
        recvRangeChange()
        {
        }

        virtual void
        recvRetry();

        virtual void
        recvRespRetry();

        virtual AddrRangeList
        getAddrRanges() const
        {
          AddrRangeList ranges;
          ranges.push_back(ownerLds->getAddrRange());
          return ranges;
        }

        template<typename T>
        void
        loadData(PacketPtr packet);

        template<typename T>
        void
        storeData(PacketPtr packet);

        template<typename T>
        void
        atomicOperation(PacketPtr packet);
    };

  protected:

    /**
     * the lds reference counter
     * The key is the workgroup ID and dispatch ID
     * The value is the number of wavefronts that reference this LDS, as
     * wavefronts are launched, the counter goes up for that workgroup and when
     * they return it decreases, once it reaches 0 then this chunk of the LDS
     * is returned to the available pool. However,it is deallocated on the 1->0
     * transition, not whenever the counter is 0 as it always starts with 0
     * when the workgroup asks for space
     */
    std::unordered_map<uint32_t,
                       std::unordered_map<uint32_t, int32_t>> refCounter;

    // the map that allows workgroups to access their own chunk of the LDS
    std::unordered_map<uint32_t,
                       std::unordered_map<uint32_t, LdsChunk>> chunkMap;

    // an event to allow the LDS to wake up at a specified time
    TickEvent tickEvent;

    // the queue of packets that are going back to the CU after a
    // read/write/atomic op
    // TODO need to make this have a maximum size to create flow control
    std::queue<std::pair<Tick, PacketPtr>> returnQueue;

    // whether or not there are pending responses
    bool retryResp = false;

    bool
    process();

    GPUDynInstPtr
    getDynInstr(PacketPtr packet);

    bool
    processPacket(PacketPtr packet);

    unsigned
    countBankConflicts(PacketPtr packet, unsigned *bankAccesses);

    unsigned
    countBankConflicts(GPUDynInstPtr gpuDynInst,
                       unsigned *numBankAccesses);

  public:
    using Params = LdsStateParams;

    LdsState(const Params &params);

    // prevent copy construction
    LdsState(const LdsState&) = delete;

    ~LdsState()
    {
        parent = nullptr;
    }

    bool
    isRetryResp() const
    {
        return retryResp;
    }

    void
    setRetryResp(const bool value)
    {
        retryResp = value;
    }

    // prevent assignment
    LdsState &
    operator=(const LdsState &) = delete;

    /**
     * use the dynamic wave id to create or just increase the reference count
     */
    int
    increaseRefCounter(const uint32_t dispatchId, const uint32_t wgId)
    {
        int refCount = getRefCounter(dispatchId, wgId);
        fatal_if(refCount < 0,
                 "reference count should not be below zero");
        return ++refCounter[dispatchId][wgId];
    }

    /**
     * decrease the reference count after making sure it is in the list
     * give back this chunk if the ref counter has reached 0
     */
    int
    decreaseRefCounter(const uint32_t dispatchId, const uint32_t wgId)
    {
      int refCount = getRefCounter(dispatchId, wgId);

      fatal_if(refCount <= 0,
              "reference count should not be below zero or at zero to"
              "decrement");

      refCounter[dispatchId][wgId]--;

      if (refCounter[dispatchId][wgId] == 0) {
        releaseSpace(dispatchId, wgId);
        return 0;
      } else {
        return refCounter[dispatchId][wgId];
      }
    }

    /**
     * return the current reference count for this workgroup id
     */
    int
    getRefCounter(const uint32_t dispatchId, const uint32_t wgId) const
    {
      auto dispatchIter = chunkMap.find(dispatchId);
      fatal_if(dispatchIter == chunkMap.end(),
               "could not locate this dispatch id [%d]", dispatchId);

      auto workgroup = dispatchIter->second.find(wgId);
      fatal_if(workgroup == dispatchIter->second.end(),
               "could not find this workgroup id within this dispatch id"
               " did[%d] wgid[%d]", dispatchId, wgId);

      auto refCountIter = refCounter.find(dispatchId);
      if (refCountIter == refCounter.end()) {
        fatal("could not locate this dispatch id [%d]", dispatchId);
      } else {
        auto workgroup = refCountIter->second.find(wgId);
        if (workgroup == refCountIter->second.end()) {
          fatal("could not find this workgroup id within this dispatch id"
                  " did[%d] wgid[%d]", dispatchId, wgId);
        } else {
          return refCounter.at(dispatchId).at(wgId);
        }
      }

      fatal("should not reach this point");
      return 0;
    }

    /*
     * return pointer to lds chunk for wgid
     */
    LdsChunk *
    getLdsChunk(const uint32_t dispatchId, const uint32_t wgId)
    {
      fatal_if(chunkMap.find(dispatchId) == chunkMap.end(),
          "fetch for unknown dispatch ID did[%d]", dispatchId);

      fatal_if(chunkMap[dispatchId].find(wgId) == chunkMap[dispatchId].end(),
          "fetch for unknown workgroup ID wgid[%d] in dispatch ID did[%d]",
          wgId, dispatchId);

      return &chunkMap[dispatchId][wgId];
    }

    bool
    returnQueuePush(std::pair<Tick, PacketPtr> thePair);

    Tick
    earliestReturnTime() const
    {
        // TODO set to max(lastCommand+1, curTick())
        return returnQueue.empty() ? curTick() : returnQueue.back().first;
    }

    void
    setParent(ComputeUnit *x_parent);

    // accessors
    ComputeUnit *
    getParent() const
    {
        return parent;
    }

    std::string
    getName()
    {
        return _name;
    }

    int
    getBanks() const
    {
        return banks;
    }

    ComputeUnit *
    getComputeUnit() const
    {
        return parent;
    }

    int
    getBankConflictPenalty() const
    {
        return bankConflictPenalty;
    }

    AddrRange
    getAddrRange() const
    {
        return range;
    }

    Port &
    getPort(const std::string &if_name, PortID idx)
    {
        if (if_name == "cuPort") {
            // TODO need to set name dynamically at this point?
            return cuPort;
        } else {
            fatal("cannot resolve the port name " + if_name);
        }
    }

    /* LDS chunks can exist in
          free (F), terminal (T), Allocated (A), and Ceded (C)
        at a given time, F+T+A+C is constant (no R there)
        a free chunk can get allocated (F->A)
        allocated can get terminal (A->T) or free (A->F)
        a terminal chunk can get reserved (R) by a lookahead Wg,
          but the chunk remains (T)
        no more reservation than available terminal chunks
        a terminal chunk can get freed if no reservations exist (T->F)
          or Ceded to reservations (T->C)
        if more terminal chunk is freed than existing reservations,
          part of it is C, and rest F
        a lookahead wg can launch with some F and some T (by reserving them)
        at a LDS resource barrier a Ceded chunk can get allocated (C->A)
    */
    int bytesFree() {
      //return maximumSize - bytesAllocated - bytesTerminal - bytesCeded;
      return _bytesFree;
    }

    void printCurrentUsage() {
      DPRINTF(GPULDS, "F %6d | T %6d ( %6d R ) "
        "| C %6d | A %6d\n",
        bytesFree(),
        bytesTerminal,
        bytesReserved,
        bytesCeded,
        bytesAllocated);
      assert (bytesTerminal + bytesAllocated <= maximumSize);
      assert (bytesTerminal >= bytesReserved);
      assert (bytesAllocated + bytesTerminal + bytesCeded + _bytesFree == maximumSize);
    }

    /**
     * can this much space be reserved for a workgroup?
     */
    bool
    canReserve(uint32_t x_size)
    {
      printCurrentUsage();
      return x_size <= bytesFree();
    }

    /**
     * assign a parent and request this amount of space be set aside
     * for this wgid
     */
    LdsChunk *
    reserveSpace(const uint32_t dispatchId,
            const uint32_t wgId, const uint32_t num_wfs,
            const uint32_t size)
    {
        printCurrentUsage();
        if (chunkMap.find(dispatchId) != chunkMap.end()) {
            panic_if(
                chunkMap[dispatchId].find(wgId) != chunkMap[dispatchId].end(),
                "duplicate workgroup ID asking for space in the LDS "
                "did[%d] wgid[%d]", dispatchId, wgId);
        }

        if (size > bytesFree()) {
            return nullptr;
        } else {
            bytesAllocated += size;
            _bytesFree -= size;

            auto value = chunkMap[dispatchId].emplace(
                        wgId, LdsChunk(size, dispatchId, wgId, num_wfs, 0));
            panic_if(!value.second, "was unable to allocate a new chunkMap");

            // make an entry for this workgroup
            refCounter[dispatchId][wgId] = 0;

            return &chunkMap[dispatchId][wgId];
        }
    }


    bool
    canEarlyReserve(uint32_t early_size, uint32_t full_size)
    {
      printCurrentUsage();

      // full_size may be entirely sat by bytesFree
      // but also by a combination of bytesFree and bytesAvailForReserv
      // bytesAvailForReserv is bytesTerminal not already reserved by others
      // early_size necessarily has to be sat by bytesFree
      return (full_size - early_size <= bytesTerminal - bytesReserved) &&
                (early_size <= bytesFree());
    }
    /**
     * assign a parent and request this amount of space be set aside
     * for this wgid
     */
    LdsChunk *
    earlyReserveSpace(const uint32_t dispatchId, const uint32_t wgId,
          const uint32_t numWfs, const uint32_t e_size,
            const uint32_t full_size)
    {
        if (chunkMap.find(dispatchId) != chunkMap.end()) {
            panic_if(
              chunkMap[dispatchId].find(wgId) != chunkMap[dispatchId].end(),
              "duplicate workgroup ID asking for space in the LDS "
              "did[%d] wgid[%d]", dispatchId, wgId);
        }

        if (!canEarlyReserve(e_size, full_size)) {
            return nullptr;
        } else {
            int reservation_size = full_size - e_size;
            int allocation_size = e_size;
            auto value = chunkMap[dispatchId].emplace(
                  wgId, LdsChunk(allocation_size, dispatchId,
                                  wgId, numWfs, reservation_size));
            panic_if(!value.second, "was unable to allocate a new chunkMap");
            bytesAllocated += allocation_size;
            _bytesFree -= allocation_size;
            bytesReserved += reservation_size;
            // make an entry for this workgroup
            refCounter[dispatchId][wgId] = 0;
            return &chunkMap[dispatchId][wgId];
        }
    }

    /* only the last wf can mark terminal or downgrade in this impl
     * this is because downgrade also waits for the last wf to downgrade
     * note: the initial WFs that fail to markTerminal (cause they're not
     * the last) will print the wrong size being attempted to markTerminal
     * the last WF will print the correct size since by then it would
     * have posted the downgrade
     */
    void
    markTerminal(uint32_t dispatchId, uint32_t wgId)
    {
      if (!chunkMap[dispatchId][wgId].isTerminal()) {
        // marking terminal first time
        DPRINTF(GPULDS, "Req to mark wgId %d terminal, size: %d\n",
          wgId, (chunkMap[dispatchId][wgId]).size());
        if (chunkMap[dispatchId][wgId].markTerminal()) {
          bytesTerminal += (chunkMap[dispatchId][wgId]).size();
          bytesAllocated -= (chunkMap[dispatchId][wgId]).size();
        } else {
          DPRINTF(GPULDS, "waiting for other WFs\n");
        }
      } else {
        DPRINTF(GPULDS, "already terminal (marked by another WF)\n");
      }
    }

    bool
    tryUpgrade(uint32_t dispatchId, uint32_t wgId)
    {
      bool can(chunkMap[dispatchId][wgId].getReservedSpace()
              <= bytesCeded);
      if (can) {
        DPRINTF(GPULDS, "can upgrade LDS for wgid %d since"
            " extension %d <= Ceded %d\n",
            wgId, chunkMap[dispatchId][wgId].getReservedSpace(),
            bytesCeded);

        // first wf in wg will service this command
        int original_size = chunkMap[dispatchId][wgId].size();
        int final_size = original_size
                     + chunkMap[dispatchId][wgId].getReservedSpace();
        DPRINTF(GPULDS, "Req to upgrade LDS for wgId %d from %d to %d\n",
            wgId, original_size, final_size);
        if (original_size < final_size) {
          bool did(chunkMap[dispatchId][wgId].upgrade());
          if (did) {
            // if actually did upgrade
            int delta = final_size - original_size;
            bytesAllocated += delta;
            bytesCeded -= delta;

            DPRINTF(GPULDS, "size of resized Wg%d %d->%d\n",
                wgId, original_size, chunkMap[dispatchId][wgId].size());

            printCurrentUsage();
            return true;
          } else {
            // else another wf already extended
            assert (false);
          }
        } else {
          // else wg prolly launched with full alloc
          DPRINTF(GPULDS, "wgId %d already has full alloc\n", wgId);
          return true;
        }
      } else {
        DPRINTF(GPULDS, "Cannot upgrade LDS for wgid %d since "
            "extension %d > ceded %d\n",
          wgId, chunkMap[dispatchId][wgId].getReservedSpace(),
          bytesCeded);
        printCurrentUsage();
        return false;
      }
    }

    // only last wf can downgrade; not before
    void
    downgrade(uint32_t dispatchId, uint32_t wgId,
      long long int pc, int32_t pct)
    {
      // make sure to never downgrade after markTerminal
      assert(!chunkMap[dispatchId][wgId].isTerminal());

      int original_size = chunkMap[dispatchId][wgId].size();
      int final_size = (int) (((float)1 - (float)pct/(float)100)
            * (float)chunkMap[dispatchId][wgId].size());
      assert (original_size >= final_size);
      DPRINTF(GPULDS, "Req to downgrade LDS for wgId %d from %d to %d\n",
        wgId, original_size, final_size);

      if (chunkMap[dispatchId][wgId].downgrade(final_size)) {
        bytesAllocated -= original_size - final_size;
        _bytesFree += original_size - final_size;

        DPRINTF(GPULDS, "Downgraded from %d to %d\n",
          original_size, chunkMap[dispatchId][wgId].size());
        printCurrentUsage();
      }

    }

  protected:
    // the number of bytes currently reserved by all workgroups
    uint32_t bytesAllocated;
    // bytes currently terminal (will soon be free)
    uint32_t bytesTerminal;
    // terminal bytes reserved
    // for a given WG, you can only extend on reserved bytes when
    // free bytes of the LDS > reserved bytes of the WG
    uint32_t bytesReserved;
    // only extend into reservable, not free
    uint32_t bytesCeded;
    uint32_t _bytesFree;

  private:
    /**
     * give back the space
     */
    bool
    releaseSpace(const uint32_t x_dispatchId, const uint32_t x_wgId)
    {
        DPRINTF(GPULDS, "LDS[%d][%d] releasing\n",
                        x_dispatchId, x_wgId);
        auto dispatchIter = chunkMap.find(x_dispatchId);

        if (dispatchIter == chunkMap.end()) {
          fatal("dispatch id not found [%d]", x_dispatchId);
        } else {
          auto workgroupIter = dispatchIter->second.find(x_wgId);
          if (workgroupIter == dispatchIter->second.end()) {
            fatal("workgroup id [%d] not found in dispatch id [%d]",
                    x_wgId, x_dispatchId);
          }
        }

        fatal_if((bytesAllocated + bytesTerminal) <
                     chunkMap[x_dispatchId][x_wgId].size(),
            "releasing more space than was allocated to the entire CU");
        int tbf = chunkMap[x_dispatchId][x_wgId].size();
        if (chunkMap[x_dispatchId][x_wgId].isTerminal()) {
          if (bytesReserved >= tbf) {
            bytesReserved -= tbf;
            bytesCeded += tbf;
            bytesTerminal -= tbf;
          } else {
            bytesCeded += bytesReserved;
            _bytesFree += tbf - bytesReserved;
            bytesTerminal -= tbf;
            bytesReserved = 0;
          }
        } else {
          bytesAllocated -= tbf;
          _bytesFree += tbf;
        }
        chunkMap[x_dispatchId].erase(chunkMap[x_dispatchId].find(x_wgId));
        printCurrentUsage();
        return true;
    }

    // the port that connects this LDS to its owner CU
    CuSidePort cuPort;

    ComputeUnit* parent = nullptr;

    std::string _name;

    // the size of the LDS, the most bytes available
    int maximumSize;

    // Address range of this memory
    AddrRange range;

    // the penalty, in cycles, for each LDS bank conflict
    int bankConflictPenalty = 0;

    // the number of banks in the LDS underlying data store
    int banks = 0;
};

} // namespace gem5

#endif // __LDS_STATE_HH__
