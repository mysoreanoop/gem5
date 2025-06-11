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
 */

#include "gpu-compute/block_register_manager_policy.hh"

#include "config/the_gpu_isa.hh"
#include "debug/GPURename.hh"
#include "debug/GPUSync.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/pool_manager.hh"
#include "gpu-compute/scalar_register_file.hh"
#include "gpu-compute/shader.hh"
#include "gpu-compute/vector_register_file.hh"
#include "gpu-compute/wavefront.hh"

namespace gem5
{
/* use of static_cast is justified because of constructor check
*/
void
BlockRegisterManagerPolicy::exec()
{
}

int
BlockRegisterManagerPolicy::mapVgpr(Wavefront* w, int vgprIndex)
{
    // this should also fire if the vgpr idx exceeds early alloc
    // but is ignored (when vgprIndex >= w->reservedVectorRegs)
    // problem is decode would require translation prior to fulfilment
    // of full vgpr requirement, no clean way to check it
    panic_if((vgprIndex >= w->maxVgprs)
             || (w->reservedVectorRegs < 0)
             || (w->reservedVectorRegs > w->maxVgprs),
             "VGPR index %d is out of range: VGPR range=[0,%d)"
             " | Wf[%d][%d] Wg%d Wf%d\n",
             vgprIndex, w->maxVgprs,
             w->simdId, w->wfSlotId, w->wgId, w->wfDynId);

    // add the offset
    return static_cast<BlockPoolManager *>
            (cu->registerManager->vrfPoolMgrs[w->simdId])->
            translate(w->vgprPtr, vgprIndex);

}

int
BlockRegisterManagerPolicy::mapSgpr(Wavefront* w, int sgprIndex)
{
    panic_if((sgprIndex >= w->reservedScalarRegs)
             || (w->reservedScalarRegs < 0)
             || (w->reservedScalarRegs > w->maxSgprs),
             "SGPR index %d is out of range: SGPR range=[0,%d)\n"
             " | Wf[%d][%d] Wg%d",
             sgprIndex, w->maxSgprs,
             w->simdId, w->wfSlotId, w->wgId);

    // add the offset
    return static_cast<BlockPoolManager *>
            (cu->registerManager->srfPoolMgrs[w->simdId])->
            translate(w->sgprPtr, sgprIndex);
}

/* for early alloc checks, it may happen that full alloc is possible
 * but when allocating, full allocation gets preferred
 */
int
BlockRegisterManagerPolicy::getTotAllocableWfsForVregUsed(
            int simdId, int demandPerWf,
            bool early, int earlyDemandPerWf) {
    DPRINTF(GPUVRF, "Checking how many WFs can be %s "
                    " mapped on SIMD%d for VGPRs used\n",
                     early ? "early" : "fully", simdId);
    // TODO compiler ensures this, but until then
    DPRINTF(GPUVRF, "demandPerWF: %d | earlyDemandPerWF: %d\n",
            demandPerWf, earlyDemandPerWf);
    assert (early? demandPerWf >= earlyDemandPerWf : true);
    if (early) {
        int out = static_cast<BlockPoolManager *>
            (cu->registerManager->vrfPoolMgrs[simdId])->
            getTotEarlyAllocableWfsForRegsUsed(demandPerWf,
                                    earlyDemandPerWf);

        DPRINTF(GPUVRF, "%d wfs early mappable\n", out);
        return out;
    } else {
        int out = static_cast<BlockPoolManager *>
            (cu->registerManager->vrfPoolMgrs[simdId])->
            getTotAllocableWfsForRegsUsed(demandPerWf);
        DPRINTF(GPUVRF, "%d wfs fully mappable\n", out);
        return out;
    }
}

int
BlockRegisterManagerPolicy::getTotAllocableWfsForSregUsed(
            int simdId, int demandPerWf,
            bool early, int earlyDemandPerWf) {
                DPRINTF(GPUVRF, "Checking how many WFs can be %s "
                    "mapped on SIMD%d for SGPRs used\n",
                     early ? "early" : "fully", simdId);
    // TODO compiler ensures this, but until then
    DPRINTF(GPUVRF, "demandPerWF: %d | earlyDemandPerWF: %d\n",
        demandPerWf, earlyDemandPerWf);
    assert (early? demandPerWf >= earlyDemandPerWf : true);
    if (early) {
        int out =  static_cast<BlockPoolManager *>
            (cu->registerManager->srfPoolMgrs[simdId])->
            getTotEarlyAllocableWfsForRegsUsed(demandPerWf,
                                    earlyDemandPerWf);
            DPRINTF(GPUVRF, "%d wfs mappable\n", out);
            return out;
    } else {
        int out = static_cast<BlockPoolManager *>
            (cu->registerManager->srfPoolMgrs[simdId])->
            getTotAllocableWfsForRegsUsed(demandPerWf);
        DPRINTF(GPUVRF, "%d wfs mappable\n", out);
        return out;

    }

}

/* marks entire region terminal */
void
BlockRegisterManagerPolicy::markTerminal(Wavefront *w) {
    DPRINTF(GPUVRF, "Marking terminal: WF[%d][%d] vgprPtr%d, size %d\n",
        w->simdId, w->wfSlotId, w->vgprPtr, w->reservedVectorRegs);
static_cast<BlockPoolManager *>
    (cu->registerManager->vrfPoolMgrs[w->simdId])->
    markRegionTerminal(w->vgprPtr, 0);

static_cast<BlockPoolManager *>
    (cu->registerManager->srfPoolMgrs[w->simdId])->
    markRegionTerminal(w->sgprPtr, 0);
}

bool
BlockRegisterManagerPolicy::canExtend(Wavefront *w)
{
    bool sgprCan = ((w->reservedScalarRegs < w->maxSgprs)
        ? static_cast<BlockPoolManager *>
            (cu->registerManager->srfPoolMgrs[w->simdId])->
            canExtend(w->sgprPtr)
        : true);

    bool vgprCan = ((w->reservedVectorRegs < w->maxVgprs)
        ? static_cast<BlockPoolManager *>
            (cu->registerManager->vrfPoolMgrs[w->simdId])->
            canExtend(w->vgprPtr)
        : true);
    if (vgprCan && sgprCan)
        DPRINTF(GPUVRF, "Wf[%d][%d] wfDynId %d can extend\n",
            w->simdId, w->wfSlotId, w->wfDynId);
    return vgprCan && sgprCan;
}

/* extend only when canExtend() */
void
BlockRegisterManagerPolicy::extendRegisters(Wavefront *w)
{
    // full allocation + zero extend allowed
    // maxVgprs and maxSgprs contain the full allocation sizes
    panic_if (w->maxVgprs < w->reservedVectorRegs,
        "vector reservation(%d) was greater than max(%d)\n",
        w->reservedVectorRegs,  w->maxVgprs);
    panic_if (w->maxSgprs < w->reservedScalarRegs,
        "scalar reservation(%d) was greater than max(%d)\n",
        w->reservedScalarRegs, w->maxSgprs);
    /* ensuring we extend by only the remaining amount
    ** after what was early allocated
    */
    if (w->maxVgprs - w->reservedVectorRegs) {
        DPRINTF(GPUVRF, "%d: Wasn't at full alloc; extending\n",
            w->wfDynId);
        DPRINTF(GPUVRF, "%d: Extending by %d VRF on WF[%d][%d]\n",
            w->wfDynId, w->maxVgprs - w->reservedVectorRegs,
            w->simdId, w->wfSlotId);

        static_cast<BlockPoolManager *>
            (cu->registerManager->vrfPoolMgrs[w->simdId])->
            extendRegion(w->vgprPtr);

        cu->vectorRegsReserved[w->simdId] +=
                    w->maxVgprs - w->reservedVectorRegs;
        w->reservedVectorRegs = w->maxVgprs;

        DPRINTF(GPUVRF, "After extending on WF[%d][%d] "
            "| curr reservation: %d\n",
            w->simdId, w->wfSlotId, cu->vectorRegsReserved[w->simdId]);

    }
    else
        DPRINTF(GPUVRF, "%d: Already at full allocation, returning\n",
                    w->wfDynId);

    if (w->maxSgprs - w->reservedScalarRegs) {
        DPRINTF(GPUVRF, "%d: Wasn't at full alloc; extending\n",
            w->wfDynId);
        DPRINTF(GPUVRF, "%d: Extending by %d SRF on WF[%d][%d]\n",
            w->wfDynId, w->maxSgprs - w->reservedScalarRegs,
            w->simdId, w->wfSlotId);

        static_cast<BlockPoolManager *>
            (cu->registerManager->srfPoolMgrs[w->simdId])->
            extendRegion(w->sgprPtr);

        cu->scalarRegsReserved[w->simdId] +=
                 w->maxSgprs - w->reservedScalarRegs;
        w->reservedScalarRegs += w->maxSgprs - w->reservedScalarRegs;

    }
    else
        DPRINTF(GPUVRF, "%d: Already at full allocation, returning\n",
                    w->wfDynId);

}

void
BlockRegisterManagerPolicy::allocateRegisters(Wavefront *w,
                int vectorDemand, int scalarDemand)
{
    DPRINTF(GPUVRF, "Allocating %d VRF on SIMD%d\n",
                    vectorDemand, w->simdId);
    uint32_t allocatedSize = 0;
    w->vgprPtr = static_cast<BlockPoolManager *>
    (cu->registerManager->vrfPoolMgrs[w->simdId])->
        allocateRegion(vectorDemand, &allocatedSize);
    w->reservedVectorRegs = allocatedSize;
    cu->vectorRegsReserved[w->simdId] += w->reservedVectorRegs;
    panic_if(cu->vectorRegsReserved[w->simdId] > cu->numVecRegsPerSimd,
             "VRF at SIMD%d has been overallocated %d > %d\n",
             w->simdId, cu->vectorRegsReserved[w->simdId],
             cu->numVecRegsPerSimd);
    DPRINTF(GPUVRF, "Allotment %d for %d regs for WF[%d][%d] "
            "| curr reservation: %d\n",
            w->vgprPtr, w->reservedVectorRegs, w->simdId,
            w->wfSlotId, cu->vectorRegsReserved[w->simdId]);
    if (scalarDemand) {
        DPRINTF(GPUVRF, "Allocating %d SRF on SIMD%d\n",
                    scalarDemand, w->simdId);

        w->sgprPtr = static_cast<BlockPoolManager *>
        (cu->registerManager->srfPoolMgrs[w->simdId])->
            allocateRegion(scalarDemand, &allocatedSize);
        w->reservedScalarRegs = allocatedSize;
        cu->scalarRegsReserved[w->simdId] += w->reservedScalarRegs;
        panic_if(cu->scalarRegsReserved[w->simdId] > cu->numScalarRegsPerSimd,
                 "SRF at SIMD%d has been overallocated %d > %d\n",
                 w->simdId, cu->scalarRegsReserved[w->simdId],
                 cu->numScalarRegsPerSimd);
        DPRINTF(GPUVRF, "Allotment %d for %d sgprs for WF[%d][%d]\n",
                 w->sgprPtr, w->reservedScalarRegs, w->simdId, w->wfSlotId);
    }

}

void
BlockRegisterManagerPolicy::allocateEarlyAndReserve(Wavefront *w,
                        int vectorDemand, int earlyVectorDemand,
                        int scalarDemand, int earlyScalarDemand)
{
    DPRINTF(GPUVRF, "Allocating %d VRF and reserving %d VRF on SIMD%d\n",
        earlyVectorDemand,
        vectorDemand - earlyVectorDemand, w->simdId);
    uint32_t grantedSize = 0;
    uint32_t reservedSize = 0;
    w->vgprPtr = static_cast<BlockPoolManager *>
        (cu->registerManager->vrfPoolMgrs[w->simdId])->
        allocateEarly(vectorDemand, earlyVectorDemand,
                &grantedSize, &reservedSize);
    w->reservedVectorRegs = grantedSize;
    cu->vectorRegsReserved[w->simdId] += w->reservedVectorRegs;
    panic_if(cu->vectorRegsReserved[w->simdId] > cu->numVecRegsPerSimd,
            "VRF[%d] has been overallocated %d > %d\n",
            w->simdId, cu->vectorRegsReserved[w->simdId],
            cu->numVecRegsPerSimd);
    DPRINTF(GPUVRF, "Allotment %d for %d regs (+reserved %d regs) "
        "for WF[%d][%d] | curr reservation: %d\n",
        w->vgprPtr, grantedSize, reservedSize, w->simdId,
        w->wfSlotId, cu->vectorRegsReserved[w->simdId]);


    if (scalarDemand) {
        grantedSize = 0;
        reservedSize = 0;

        DPRINTF(GPUVRF,
            "Allocating %d SRF and reserving %d SRF on SIMD%d\n",
            earlyScalarDemand,
            scalarDemand - earlyScalarDemand, w->simdId);
        w->sgprPtr = static_cast<BlockPoolManager *>
            (cu->registerManager->srfPoolMgrs[w->simdId])->
            allocateEarly(scalarDemand, earlyScalarDemand,
                &grantedSize, &reservedSize);
        w->reservedScalarRegs = grantedSize;
        cu->scalarRegsReserved[w->simdId] += w->reservedScalarRegs;
        panic_if(cu->scalarRegsReserved[w->simdId] > cu->numScalarRegsPerSimd,
                "SRF[%d] has been overallocated %d > %d\n",
                w->simdId, cu->scalarRegsReserved[w->simdId],
                cu->numScalarRegsPerSimd);
        DPRINTF(GPUVRF, "Allocated %d SRF and reserved %d VRF on SIMD%d\n",
            grantedSize, reservedSize, w->simdId);

    }

}

void
BlockRegisterManagerPolicy::freeRegisters(Wavefront *w) {
    partialFreeRegisters(w, w->reservedVectorRegs, w->reservedScalarRegs);
}

void
BlockRegisterManagerPolicy::partialFreeRegisters(Wavefront *w,
            int vgprsToFree, int sgprsToFree)
{
    DPRINTF(GPUSync, "Freeing %d(%d|%d) VGPRs and %d(%d) SGPRs\n"
                     "WF[%d][%d] %d allocation: %d, size %d\n",
                        vgprsToFree, w->reservedVectorRegs,
                        w->computeUnit->vectorRegsReserved[w->simdId],
                        sgprsToFree, w->reservedScalarRegs,
                        w->simdId, w->wfSlotId, w->wfDynId,
                        w->vgprPtr, w->reservedVectorRegs
                    );

    if (vgprsToFree > 0) {
        // free the vector registers of the completed wavefront
        assert(vgprsToFree <= w->reservedVectorRegs);
        w->computeUnit->vectorRegsReserved[w->simdId] -= vgprsToFree;
        // TODO make sure the remaining VGPRs still >= grain size (16?)
        panic_if(w->computeUnit->vectorRegsReserved[w->simdId] < 0,
                 "VRF[%d]: %d registers reserved\n",
                 w->simdId,
                 w->computeUnit->vectorRegsReserved[w->simdId]);

        // mark/pre-mark all registers are not busy
        DPRINTF(GPUVRF, "About to mark VGPRs false\n");
        for (int i = w->reservedVectorRegs - vgprsToFree;
                                i < w->reservedVectorRegs; i++) {
            uint32_t physVgprIdx = mapVgpr(w, i);
            w->computeUnit->vrf[w->simdId]->markReg(physVgprIdx, false);
        }
        static_cast<BlockPoolManager *>
            (w->computeUnit->registerManager->vrfPoolMgrs[w->simdId])->
            freeRegion(w->vgprPtr, w->reservedVectorRegs - vgprsToFree);


        w->reservedVectorRegs -= vgprsToFree;
        DPRINTF(GPUVRF, "After freeing %d regs on WF[%d][%d] "
            "| curr reservation: %d\n",
            vgprsToFree, w->simdId, w->wfSlotId,
            cu->vectorRegsReserved[w->simdId]);

    }

    if (sgprsToFree > 0) {
        // free the scalar registers of the completed wavefront
        assert(sgprsToFree <= w->reservedScalarRegs);
        w->computeUnit->scalarRegsReserved[w->simdId] -= sgprsToFree;
        panic_if(w->computeUnit->scalarRegsReserved[w->simdId] < 0,
                 "SRF[%d]: %d registers reserved\n",
                 w->simdId,
                 w->computeUnit->scalarRegsReserved[w->simdId]);

        // mark/pre-mark all registers are not busy
        DPRINTF(GPUVRF, "About to mark SGPRs false\n");
        for (int i = w->reservedScalarRegs - sgprsToFree;
                                i < w->reservedScalarRegs; i++) {
            uint32_t physSgprIdx = mapSgpr(w, i);
            w->computeUnit->srf[w->simdId]->markReg(physSgprIdx, false);
        }
        static_cast<BlockPoolManager *>
            (w->computeUnit->registerManager->srfPoolMgrs[w->simdId])->
            freeRegion(w->sgprPtr, w->reservedScalarRegs - sgprsToFree);

        w->reservedScalarRegs -= sgprsToFree;
    }

    DPRINTF(GPUSync, "After: %d->%d VGPRs and %d->%d SGPRs\n"
        "WF[%d][%d] allocation: [%d, +%d)\n",
           vgprsToFree, w->reservedVectorRegs,
           sgprsToFree, w->reservedScalarRegs,
           w->simdId, w->wfSlotId, w->vgprPtr, w->reservedVectorRegs
       );
}

} // namespace gem5
