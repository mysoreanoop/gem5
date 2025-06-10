/*
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include "gpu-compute/block_pool_manager.hh"

#include <algorithm>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/GPUVRF.hh"

namespace gem5
{



// reset freeSpace and reservedSpace
void
BlockPoolManager::resetRegion(const int & regsPerSimd){
    totalRegSpace = regsPerSimd;

    // reset available free space
    _totRegSpaceAvailable = regsPerSimd;
}

// calculates number of blocks needed for a given size
// guaranteed to be at least 1
int BlockPoolManager::calculateNumBlocksNeeded(const uint32_t size) const {
    assert (size > 0 && size < m_poolSize);
    // ceiling division
    return divCeil(size, m_blockSize);
}

/* count blocks with a specific status;
 design choice instead of maintaining separate counters
 for blocks for the 4 possible statuses
*/
int BlockPoolManager::countBlocksWithStatus(BlockStatus status) const {
    int count = 0;
    for (const auto& block : m_pool) {
        if (block.status == status) {
            count++;
        }
    }
    return count;
}

// gets indices of blocks overlapping a given address range
std::vector<int> inline BlockPoolManager::getBlockIdsInRange(
            uint32_t virtualStart, uint32_t virtualEnd,
            AllocationRecord& record) const {
    std::vector<int> indices;
    uint32_t currentVirtualBase = 0;
    assert (virtualEnd > virtualStart);
    for (int b : record.grantedBlocks) {
        assert(checkBId(b));

        // calculate virtual range covered by this physical block
        uint32_t bVirtualStart = currentVirtualBase;

        // check for overlap between the block's physical range
        // [bVirtualStart, bVirtualEnd)
        // and the target virtual range [virtualStart, virtualEnd)
        if (bVirtualStart >= virtualStart &&
                bVirtualStart < virtualEnd) {
            // mark this block index for release
            indices.push_back(b);
        }

        // move to the next block's virtual starting point
        currentVirtualBase += m_blockSize;
    }

    return indices;
}

// finds 'count' blocks matching status
std::vector<int> BlockPoolManager::findBlocks(int count,
            BlockStatus requiredStatus) const {
    std::vector<int> bIds;
    if (count == 0) return bIds; // possible
    bIds.reserve(count);
    for (int i = 0; i < m_pool.size() && bIds.size() < count; ++i) {
        const auto& block = m_pool[i];
        if (block.status == requiredStatus) {
            bIds.push_back(i);
        }

    }
    return bIds;
}

// check how many instances of size can be allocated
int BlockPoolManager::getTotAllocableWfsForRegsUsed(uint32_t size) const {
    // check if enough FREE blocks exist
    return countBlocksWithStatus(BlockStatus::FREE) /
            calculateNumBlocksNeeded(size);
}

// check how many instances of (earlySize/fullSize) can be allocated/reserved
int BlockPoolManager::getTotEarlyAllocableWfsForRegsUsed(uint32_t fullSize,
        uint32_t earlySize) const {
    assert (fullSize > earlySize);

    int minBlocksPerRequest = calculateNumBlocksNeeded(earlySize);
    int extBlocksPerRequest = calculateNumBlocksNeeded(fullSize-earlySize);

    // Count available resources
    int freeBlockCount = 0;
    int terminalBlockCount = 0;
    for (const auto& block : m_pool) {
        if (block.status == BlockStatus::FREE) {
            freeBlockCount++;
        } else if (block.status == BlockStatus::TERMINAL) {
            // Count terminal blocks that are not currently reserved
            terminalBlockCount++;
        }
    }

    // Check if total free and total terminal can be met
    return std::min(freeBlockCount / minBlocksPerRequest,
            terminalBlockCount / extBlocksPerRequest);
}


// allocates an instance of specified size
uint32_t BlockPoolManager::allocateRegion(
            const uint32_t size, uint32_t *grantedPoolSize) {
                int numBlocksNeeded = calculateNumBlocksNeeded(size);
    DPRINTF(GPUVRF, "Allocating region; num blocks needed: %d\n",
            numBlocksNeeded);
    /* assumption: between calling canAllocate and allocate,
        there cannot be interceding requests, the caller only calls allocate
        if canAllocate returns true
    */

    // find the required number of free blocks
    std::vector<int> freeBlocks = findBlocks(numBlocksNeeded,
                                            BlockStatus::FREE);
    if (freeBlocks.size() < numBlocksNeeded) {
        DPRINTF(GPUVRF, "Only call allocate for no more than "
            "as many blocks as can be allocated\n");
        *grantedPoolSize = 0;
        return NO_OWNER_ID;
    }

    // allocation
    AllocationRecord record(nextId, size);
    *grantedPoolSize = numBlocksNeeded * m_blockSize;
    for (int blockIndex : freeBlocks) {
        m_pool[blockIndex].alloc(BlockStatus::GRANTED, nextId, NO_OWNER_ID);
        record.grantedBlocks.push_back(blockIndex);
    }

    m_activeAllocations.emplace(nextId, std::move(record));

    DPRINTF(GPUVRF, "Allocated region at %d, size %d; num blocks: %d\n",
        m_pool[freeBlocks[0]].startAddress, *grantedPoolSize,
        numBlocksNeeded);

    return nextId++;
}

uint32_t BlockPoolManager::allocateEarly(
        uint32_t fullSize, uint32_t earlySize,
        uint32_t *grantedPoolSize, uint32_t *reservedPoolSize) {
    /* assumption: canAllocateEarly could potentially be fully allocated
        CU first tries to allocate fully, failing which comes here
    */
   int minBlocksNeeded = calculateNumBlocksNeeded(earlySize);
   int extBlocksNeeded = (earlySize == fullSize) ? 0 :
                        calculateNumBlocksNeeded(fullSize-earlySize);
    /* TODO corner case possible to accommodate within minBlocksNeeded
        if internal remainder fragment suffices
    */

    // find blocks
    std::vector<int> freeBlocks =
                    findBlocks(minBlocksNeeded, BlockStatus::FREE);
    std::vector<int> terminalBlocks;

    // TODO opt: don't restrict to terminal blocks
    if (extBlocksNeeded)
        terminalBlocks = findBlocks(extBlocksNeeded, BlockStatus::TERMINAL);

    if (freeBlocks.size() < minBlocksNeeded
        || terminalBlocks.size() < extBlocksNeeded) {
        panic("Only call allocateEarly for no more than "
                "as many blocks as can be allocated\n");
    }

    // record only early size for now, update when extending
    AllocationRecord record(nextId, earlySize);

    // allocate FREE blocks
    for (int blockIndex : freeBlocks) {
        m_pool[blockIndex].alloc(BlockStatus::GRANTED, nextId, NO_OWNER_ID);
        record.grantedBlocks.push_back(blockIndex);
    }
    *grantedPoolSize = minBlocksNeeded * m_blockSize;

    // reserve TERMINAL blocks
    if (extBlocksNeeded) {
        for (int blockIndex : terminalBlocks) {
            // status turns to RESERVED, owner remains original
            m_pool[blockIndex].reserve(nextId); // mark who reserved it
            record.reservedBlocks.push_back(blockIndex);
        }
        *reservedPoolSize = extBlocksNeeded * m_blockSize;
    }

    m_activeAllocations.emplace(nextId, std::move(record));
    DPRINTF(GPUVRF, "Allocated region at %d, size %d; num blocks %d\n",
        m_pool[freeBlocks[0]].startAddress, *grantedPoolSize,
        minBlocksNeeded);
    if (extBlocksNeeded)
        DPRINTF(GPUVRF, "Reserved region at %d, size %d, num blocks: %d\n",
            m_pool[terminalBlocks[0]].startAddress, reservedPoolSize,
            extBlocksNeeded);


    return nextId++;
}


// handles state transitions for one block when freeing
void inline BlockPoolManager::processBlockRelease(int bId) {
    assert(checkBId(bId));

    RegisterBlock& block = m_pool[bId];

    if (block.status == BlockStatus::GRANTED ||
            block.status == BlockStatus::TERMINAL) {
        block.alloc(BlockStatus::FREE, NO_OWNER_ID, NO_OWNER_ID);
    } else if (block.status == BlockStatus::RESERVED) {
        // some other wf has reserved this region
        // presumably it was terminal before that
        // yield the block to the reserving wf
        // and clear the reservation
        block.alloc(BlockStatus::GRANTED, block.reserverId, NO_OWNER_ID);
    }
}

/* free a subset of blocks identified by
    avirtual address range within the requestor's view
*/
void BlockPoolManager::freeRegion(uint32_t id, uint32_t virtualStart) {
    // 1. Find the allocation record
    auto allocIt = m_activeAllocations.find((int)id);
    if (allocIt == m_activeAllocations.end()) {
        printStatus();
        fatal("Freeing unknown allocation\n");
    }
    AllocationRecord& record = allocIt->second;
    assert (virtualStart < record.allocationSize);
    DPRINTF(GPUVRF, "Freeing alloc %d, allocation size %d, starting at %d\n",
        id,  record.allocationSize, virtualStart);
    /* ensures the range ends at the allocation's end
        because otherwise untargeted blocks which would
        span a virtually continuous range, will reorder
        register virtualId to physicalId translations
    */

    // identify block indices within the virtual range
    std::vector<int> blocksToRelease =
        getBlockIdsInRange(virtualStart, record.allocationSize, record);

    // cannot have no blocks to release at this point
    assert (!blocksToRelease.empty());

    // process the release for the identified blocks and update the record
    // TODO use function pointer as an argument to processBlockRelease for this
    // and markRangeAsTerminal can reuse this
    for (int b : blocksToRelease) {
        // this block was identified for release, process it
        processBlockRelease(b);
    }

    // update alloc record
    if (record.grantedBlocks.size() == blocksToRelease.size()) {
        // Remove the record entirely
        m_activeAllocations.erase(allocIt);

    } else {
        // free partially
        record.grantedBlocks.resize(
                record.grantedBlocks.size()-blocksToRelease.size());
        // record.reservedBlocks remains unchanged
        record.allocationSize -= blocksToRelease.size() * m_blockSize;

    }
}


// marks a range within a GRANTED allocation as TERMINAL
bool BlockPoolManager::markRegionTerminal(int id, uint32_t virtualStart) {
    auto allocIt = m_activeAllocations.find(id);
    if (allocIt == m_activeAllocations.end()) {
        printStatus();
        fatal("Marking unknown allocation\n");
    }
    AllocationRecord& record = allocIt->second;

    /* TODO not sure if range needs to be restricted
        don't see a situation where intermediate
        ranges can be marked terminal without
        major overhaul
    */
    std::vector<int> indicesInRange;
    if (virtualStart != 0)
        std::vector<int> indicesInRange =
            getBlockIdsInRange(virtualStart, record.allocationSize, record);
    else
        indicesInRange = record.grantedBlocks;
    // cannot have no blocks to mark at this point
    assert (!indicesInRange.empty());

    for (int b : indicesInRange) {
        // check if the block index is in the granted set for this request ID
        m_pool[b].mark(BlockStatus::TERMINAL);
        assert (m_pool[b].reserverId == NO_OWNER_ID);
    }

    return true;
}


bool BlockPoolManager::canExtend(int id) {
    auto allocIt = m_activeAllocations.find(id);
    if (allocIt == m_activeAllocations.end()) {
        printStatus();
        fatal("Querying unknown allocation %d\n", id);
    }
    AllocationRecord& record = allocIt->second;

    bool can(true);
    int cannotBId = -2; // cause -1 is for default bId
    for (int bId : record.reservedBlocks) {
        bool check(m_pool[bId].status == BlockStatus::GRANTED);
                // do not check for reserverId match cause original
                // owner would have reset it when freeing its
                // terminal-turned-reserved block

        can &= check;
        if (!check) {
            assert (m_pool[bId].status == BlockStatus::RESERVED);
            cannotBId = bId;
        }
    }
    if (!can) {
        DPRINTF(GPUVRF, "cannot extend because block %d "
            "of alloc %d is not yet relinquished by alloc %d\n",
            cannotBId, id, m_pool[cannotBId].ownerId);
    }
    return can;
}

void BlockPoolManager::extendRegion(int id) {
    assert (canExtend(id));

    auto allocIt = m_activeAllocations.find(id);
    if (allocIt == m_activeAllocations.end()) {
        printStatus();
        fatal("Extending unknown allocation\n");
    }
    AllocationRecord& record = allocIt->second;

    // update the allocation size
    record.allocationSize += m_blockSize * record.reservedBlocks.size();
    DPRINTF(GPUVRF, "Extending: Updated allocation size to %d\n",
        record.allocationSize);
    record.grantedBlocks.insert(record.grantedBlocks.end(),
            record.reservedBlocks.begin(),
            record.reservedBlocks.end());
    // reset the reserverdBlocks since granted now
    record.reservedBlocks.clear();
}


// translate virtual offset to physical address
uint32_t BlockPoolManager::translate(int id,
                            uint32_t virtualOffset) const {
    // find the allocation record
    auto allocIt = m_activeAllocations.find(id);
    if (allocIt == m_activeAllocations.end()) {
        printStatus();
        fatal("Translating index for unknown allocation\n");
    }
    const AllocationRecord& record = allocIt->second;

    if (virtualOffset < record.allocationSize) {
        // falls within GRANTED blocks
        uint32_t currentVirtualBase = 0;
        for (int bId : record.grantedBlocks) {
            assert(checkBId(bId));
            if (virtualOffset >= currentVirtualBase &&
                    virtualOffset < currentVirtualBase + m_blockSize) {
                // found the correct physical block
                const RegisterBlock& b = m_pool[bId];

                // ensure the block is not free
                if (b.status == BlockStatus::FREE ||
                         b.ownerId != record.id) {
                    record.print();
                    printStatus();
                    printf("Block free or not owned; "
                    "owner: %d, extpected %d, index %d\n",
                    b.ownerId, record.id, virtualOffset);
                    fatal("translate failed\n");
                }

                // calculate the offset within this specific physical block
                uint32_t offsetInBlock = virtualOffset - currentVirtualBase;

                // calculate and return the absolute physical address
                return b.startAddress + offsetInBlock;
            }

            // move to the next block's virtual range start
            currentVirtualBase += m_blockSize;
        }
    } else {
        /* offset in reservedBlocks & may not be granted yet;
            happens when translation requests arrive before
            allocation extends; so translate proactively
            assuming use of the xlated reg is blocked by the
            resource barrier
        */
        // start from after granted blocks
        uint32_t currentVirtualBase = 0;
        std::vector<int> combinedBlocks;
        combinedBlocks.reserve(record.grantedBlocks.size() +
                        record.reservedBlocks.size());
        combinedBlocks.insert(combinedBlocks.end(),
                        record.grantedBlocks.begin(),
                        record.grantedBlocks.end());
        combinedBlocks.insert(combinedBlocks.end(),
                        record.reservedBlocks.begin(),
                        record.reservedBlocks.end());

        for (int bId : combinedBlocks) {
           assert(checkBId(bId));
           if (virtualOffset >= currentVirtualBase &&
                     virtualOffset < currentVirtualBase + m_blockSize) {
               // Found the correct physical block
               const RegisterBlock& b = m_pool[bId];

                // Ensure the block is either not free and owned
                // Or terminal and reserved for
                if (!((b.status != BlockStatus::FREE && b.ownerId == record.id)
                        || (b.status == BlockStatus::RESERVED &&
                        b.reserverId == record.id))
                    ) {
                    record.print();
                    printStatus();
                    printf("Block either not free and owned, "
                        "or, reserved by not this requestor;"
                        "owner %d, requested by %d, index %d\n",
                    b.ownerId, record.id, virtualOffset);
                    fatal("translate failed\n");
                }
               // calculate the offset within this specific physical block
               uint32_t offsetInBlock = virtualOffset - currentVirtualBase;

               // calculate and return the absolute physical address
               return b.startAddress + offsetInBlock;
           }

           // move to the next block's virtual range start
           currentVirtualBase += m_blockSize;
        }
    }
    printStatus();
    // offset out of range of even reserved blocks
    fatal ("translation of virt to phy vgpr/sgpr failed\n");
}

void BlockPoolManager::printStatus() const {
    std::stringstream ss;

    ss << "\n--- Register Pool Status ---" << "\n";

    // adjust widths for abbreviated status
    const int idxW = 5;
    const int addrW = 10; // "0x" + 16 hex digits
    const int statusW = 4;
    const int ownerW = 10;
    const int reserverW = 12;
    const int totalW = idxW + addrW + statusW + ownerW + reserverW;

    ss << std::left << std::setw(idxW) << "Idx"
              << std::setw(addrW) << "Start Addr"
              << std::setw(statusW) << "St"
              << std::setw(ownerW) << "OwnerID"
              << std::setw(reserverW) << "ReserverID" << "\n";
     ss << std::string(totalW, '-') << "\n";

    for (int i = 0; i < m_pool.size(); ++i) {
        if (m_pool[i].status != BlockStatus::FREE) {
            const auto& block = m_pool[i];
            ss << std::left << std::setw(idxW) << i
                    << std::setfill(' ')
                    << std::setw(addrW)
                    << block.startAddress
                    << std::setfill(' ') // pad address with 0s
                    << std::setw(statusW); // switch back to decimal for status

            switch (block.status) {
                case BlockStatus::FREE:     ss << "F"; break;
                case BlockStatus::GRANTED:  ss << "G"; break;
                case BlockStatus::TERMINAL: ss << "T"; break;
                case BlockStatus::RESERVED: ss << "R"; break;
            }

            ss << std::setw(ownerW); // owner ID column
            if (block.ownerId == NO_OWNER_ID)
                ss << "-";
            else
                ss << block.ownerId;

            ss << std::setw(reserverW); // reserved ID column
            if (block.reserverId == NO_OWNER_ID)
                ss << "-";
            else
                ss << block.reserverId;

            ss << "\n";
        }
    }
    ss << "---------------------------\n";
    ss << "\n--- Active Allocations ---\n";
    DPRINTF(GPUVRF, "%s", ss.str());
    if (!m_activeAllocations.empty()) {
        for (const auto& pair : m_activeAllocations) {
            pair.second.print();
        }
    }

}


} // namespace gem5
