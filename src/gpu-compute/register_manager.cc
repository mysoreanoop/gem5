/*
* Copyright (c) 2016, 2017 Advanced Micro Devices, Inc.
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
* Author: Mark Wyse
*/

#include "gpu-compute/register_manager.hh"
#include "config/the_gpu_isa.hh"
#include "debug/GPURename.hh"
#include "gpu-compute/block_register_manager_policy.hh"
#include "gpu-compute/compute_unit.hh"
#include "gpu-compute/scalar_register_file.hh"
#include "gpu-compute/static_register_manager_policy.hh"
#include "gpu-compute/vector_register_file.hh"
#include "gpu-compute/wavefront.hh"
#include "params/RegisterManager.hh"

namespace gem5
{

RegisterManager::RegisterManager(const RegisterManagerParams &p)
    : SimObject(p), srfPoolMgrs(p.srf_pool_managers),
      vrfPoolMgrs(p.vrf_pool_managers)
{
    if (p.policy == "static") {
        policy = new StaticRegisterManagerPolicy();
    } else if (p.policy == "block") {
        policy = new BlockRegisterManagerPolicy();
    } else {
        fatal("Register Manager policy not recognized;"
            "`block` or `static` only please\n");
    }

}

RegisterManager::~RegisterManager()
{
    for (auto mgr : srfPoolMgrs) {
        delete mgr;
    }
    for (auto mgr : vrfPoolMgrs) {
        delete mgr;
    }
}

void
RegisterManager::exec()
{
    policy->exec();
}

void
RegisterManager::setParent(ComputeUnit *cu)
{
    computeUnit = cu;
    policy->setParent(computeUnit);
    for (int i = 0; i < srfPoolMgrs.size(); i++) {
        fatal_if(computeUnit->srf[i]->numRegs() %
                 srfPoolMgrs[i]->minAllocation(),
                 "Min SGPR allocation is not multiple of VRF size\n");
    }
    for (int i = 0; i < vrfPoolMgrs.size(); i++) {
        fatal_if(computeUnit->vrf[i]->numRegs() %
                 vrfPoolMgrs[i]->minAllocation(),
                 "Min VGPG allocation is not multiple of VRF size\n");
    }
}

// compute mapping for vector register
int
RegisterManager::mapVgpr(Wavefront* w, int vgprIndex)
{
    return policy->mapVgpr(w, vgprIndex);
}

// compute mapping for scalar register
int
RegisterManager::mapSgpr(Wavefront* w, int sgprIndex)
{
    return policy->mapSgpr(w, sgprIndex);
}

// check if we can allocate registers
bool
RegisterManager::canAllocateVgprs(int simdId, int nWfs, int demandPerWf)
{
    return policy->canAllocateVgprs(simdId, nWfs, demandPerWf);
}

int
RegisterManager::getTotAllocableWfsForVregUsed(
        int simdId, int demandPerWf, bool early, int earlyDemandPerWf)
{
   return policy->getTotAllocableWfsForVregUsed(
                        simdId, demandPerWf, early, earlyDemandPerWf);
}

int
RegisterManager::getTotAllocableWfsForSregUsed(
        int simdId, int demandPerWf, bool early, int earlyDemandPerWf)
{
    return policy->getTotAllocableWfsForSregUsed(
        simdId, demandPerWf, early, earlyDemandPerWf);
}


bool
RegisterManager::canAllocateSgprs(int simdId, int nWfs, int demandPerWf)
{
    return policy->canAllocateSgprs(simdId, nWfs, demandPerWf);
}

// allocate registers
void
RegisterManager::allocateRegisters(Wavefront *w, int vectorDemand,
                                   int scalarDemand)
{
    policy->allocateRegisters(w, vectorDemand, scalarDemand);
}

void
RegisterManager::allocateEarlyAndReserve(Wavefront *w,
    int vectorDemand, int earlyVectorDemand,
    int scalarDemand, int earlyScalarDemand)
{
    if (BlockRegisterManagerPolicy* blockPolicy
            = dynamic_cast<BlockRegisterManagerPolicy*>(policy)) {
        blockPolicy->
            allocateEarlyAndReserve(w, vectorDemand, earlyVectorDemand,
            scalarDemand, earlyScalarDemand);
    } else {
        panic ("Method %s can only be invoked by `block` policy\n",
                __func__);
    }
}

void
RegisterManager::freeRegisters(Wavefront* w)
{
    policy->freeRegisters(w);
}

void
RegisterManager::partialFreeRegisters(Wavefront* w, int vgprs, int sgprs)
{
    if (BlockRegisterManagerPolicy* blockPolicy =
            dynamic_cast<BlockRegisterManagerPolicy*>(policy)) {
        blockPolicy->partialFreeRegisters(w, vgprs, sgprs);
    } else {
        panic ("Method %s can only be invoked by `block` policy\n",
                __func__);
    }
}

void RegisterManager::markTerminal(Wavefront *w)
{
    if (BlockRegisterManagerPolicy* blockPolicy =
            dynamic_cast<BlockRegisterManagerPolicy*>(policy)) {
        blockPolicy->markTerminal(w);
    } else {
        panic ("Method %s can only be invoked by `block` policy\n",
                __func__);
    }
}
bool RegisterManager::canExtend(Wavefront *w)
{
    if (BlockRegisterManagerPolicy* blockPolicy =
            dynamic_cast<BlockRegisterManagerPolicy*>(policy)) {
        return blockPolicy->canExtend(w);
    } else {
        panic ("Method %s can only be invoked by `block` policy\n",
                __func__);
    }
}
void RegisterManager::extendRegisters(Wavefront *w)
{
    if (BlockRegisterManagerPolicy* blockPolicy =
            dynamic_cast<BlockRegisterManagerPolicy*>(policy)) {
        blockPolicy->extendRegisters(w);
    } else {
        panic ("Method %s can only be invoked by `block` policy\n",
                __func__);
    }
}

} // namespace gem5
