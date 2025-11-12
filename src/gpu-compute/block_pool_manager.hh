/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
 *
 */

#ifndef __BLOCK_POOL_MANAGER_HH__
#define __BLOCK_POOL_MANAGER_HH__

#include <cassert>
#include <cstdint>
#include <iomanip>  // for printStatus formatting
#include <map>
#include <stdexcept> // for exceptions
#include <string>
#include <vector>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/GPUVRF.hh"
#include "gpu-compute/pool_manager.hh"
#include "params/BlockPoolManager.hh"

namespace gem5
{
    // status of a register block
    enum class BlockStatus
    {
        FREE,      // available for allocation
        GRANTED,   // allocated to a requestor
        TERMINAL,  // marked for imminent release (reservable)
        RESERVED   // released by original owner;
                   // reserved for a specific waiting requestor
    };

    // block not owned
    const int NO_OWNER_ID = -1;

    // register block
    struct RegisterBlock
    {
        const uint32_t startAddress;
        BlockStatus status = BlockStatus::FREE;
        int ownerId = NO_OWNER_ID;
        /* ID of the requestor this block is reserved for
        * set when a TERMINAL block is reserved via allocateEarly
        */
        int reserverId = NO_OWNER_ID;

        RegisterBlock(uint32_t addr) : startAddress(addr) {}

        // use for both alloc and free
        void alloc(BlockStatus s, int oId, int rId) {
            status = s; ownerId = oId; reserverId = rId;
        }
        void reserve(int rId) {
            status = BlockStatus::RESERVED; reserverId = rId;
        }
        void mark(BlockStatus s) {
            status = s;
        }
    };

    // active allocation (and reservation)
    struct AllocationRecord
    {
        int id = NO_OWNER_ID; // allocator ID
        uint32_t allocationSize; // original requested size
        // IDs of blocks currently GRANTED for this request
        std::vector<int> grantedBlocks;
        // IDs of blocks currently TERMINAL/RESERVED
        std::vector<int> reservedBlocks;

        AllocationRecord(int id, uint32_t size) :
            id(id), allocationSize(size) {}

        void print() const {
            std::string granted_str;
            for (size_t idx : grantedBlocks) {
                granted_str += std::to_string(idx) + " ";
            }
            std::string reserved_str;
            for (size_t idx : reservedBlocks) {
                reserved_str += std::to_string(idx) + " ";
            }

            DPRINTF(GPUVRF, "Record[%d]: size(%d), numGranted(%d), "
                "numReserved(%d)\n\tgranted:[%s]\n\treserved[%s]\n",
                id, allocationSize, grantedBlocks.size(),
                reservedBlocks.size(), granted_str, reserved_str);
        }
    };


/* BlockPoolManager allocates VGPRs in the granularity of m_blockSize
* non-contiguously, reducing external fragmentation;
* maintaines allocation record or individual wfs;
* each record contains the VGPR blocks in possession
*/
class BlockPoolManager : public PoolManager
{
  public:
    BlockPoolManager(const PoolManagerParams &p)
        : PoolManager(p), _regionSize(0)
    {
        if (p.min_alloc == 0) {
            throw std::invalid_argument("block size cannot be zero.");
        }
        m_poolSize = p.pool_size;
        m_blockSize = p.min_alloc;
        totalRegSpace = _totRegSpaceAvailable = p.pool_size;

        // ceiling division
        int numBlocks = (p.pool_size + p.min_alloc - 1) / p.min_alloc;
        m_pool.reserve(numBlocks);

        for (int i = 0; i < numBlocks; ++i) {
            // blocks start at address 0 and are contiguous
            m_pool.emplace_back((uint32_t)i * p.min_alloc);
        }
    }

    // check how many instances of a full alloc reqs can be satisfied
    int getTotAllocableWfsForRegsUsed(uint32_t size);

    // check how many instances of earlySize FREE blocks
    // and fullSize TERMINAL blocks can be satisfied
    int getTotEarlyAllocableWfsForRegsUsed(
            uint32_t fullSize, uint32_t earlySize) const;

    // allocate blocks for a given size
    // returns an allocation ID if successful
    uint32_t allocateRegion(const uint32_t size,
                uint32_t *grantedPoolSize) override;

    // early/partial allocation: allocate earlySize in FREE blocks,
    // reserve the difference in sizes in TERMINAL blocks
    // returns the request ID if successful.
    uint32_t allocateEarly(const uint32_t fullSize, uint32_t earlySize,
                uint32_t *grantedPoolSize, uint32_t *reservedPoolSize);

    // relinquish only blocks conatining addr
    // starting from (including) specified addr
    void freeRegion(uint32_t id, uint32_t virtualStart) override;

    // mark a range of currently granted virtual addresses as TERMINAL
    bool markRegionTerminal(int id, uint32_t virtualStart);

    /* checks if full allocation has been granted;
        this would happen silently when the original owner
        of reserved blocks relinquishes its blocks,
        which will reassociate reserved blocks to their
        reserving owners through processBlockRelease();
        obviates needing to explicitly extend allocations
    */
    bool canExtend(int id);
    void extendRegion(int id);

    // overridden
    std::string printRegion() override { return "";}
    bool canAllocate(uint32_t numRegions, uint32_t size) override
                                      { return false; }
    uint32_t regionSize(std::pair<uint32_t,uint32_t> &region) override
                                      { return 0; }
    void resetRegion(const int & regsPerSimd) override;

    // translate a virtual offset to an absolute physical address.
    uint32_t translate(int id, uint32_t virtualOffset) const;

  private:
    // actual size of a region (normalized to the minimum size that can
    // be reserved)
    uint32_t _regionSize;
    // total registers available - across chunks
    uint32_t _totRegSpaceAvailable;

    int totalRegSpace;
    int nextId = 0;
    uint32_t m_blockSize;
    uint32_t m_poolSize;

    /* utility functions */
    int countBlocksWithStatus(BlockStatus status) const;
    int calculateNumBlocksNeeded(const uint32_t size) const;
    void printStatus() const;
    bool checkBId(int bId) const {
        if (bId < m_pool.size())
            return true;
        else {
            DPRINTF(GPUVRF, "bId %d exceeds pool size, %d\n",
                    bId, m_pool.size());
            printStatus();
            return false;
        }
    };

    /* current design maintains a pool of contiguous blocks
        this was a design choice for better cache locality for
        efficient scanning compared to storing potentially
        scattered block pointers within each allocation record
    */
    std::vector<RegisterBlock> m_pool;
    // map key is request ID
    std::map<uint32_t, AllocationRecord> m_activeAllocations;
    /* helper functions */
    std::vector<int> findBlocks(int count, BlockStatus requiredStatus) const;
    std::vector<int> inline getBlockIdsInRange(
        uint32_t virtualStart, uint32_t length,
            AllocationRecord& record) const;
    void inline processBlockRelease(int bId);

};

} // namespace gem5

#endif // __BLOCK_POOL_MANAGER_HH__
