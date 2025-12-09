/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include "ocs2_mpc/MPC_MRT_Interface.h"

#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_core/control/LinearController.h>

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MPC_MRT_Interface::MPC_MRT_Interface(MPC_BASE& mpc, ::ros::NodeHandle nodeHandle) : mpc_(mpc) {
  mpcTimer_.reset();

  // // subscribe with member callback
  m_ocs_caches.resize(16);

  int i=0;
  for(auto& cache: m_ocs_caches){
    int cache_size = 64;
    if(i==6 || i==9)
      cache_size=512;
    cache = std::make_unique<MPCCache_ocs>(cache_size, 0.5);
    i++;
  }
  cmdVelSub_ = nodeHandle.subscribe ("/cmd_vel", 1, &MPC_MRT_Interface::cmdVelCallback, this);

}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::resetMpcNode(const TargetTrajectories& initTargetTrajectories) {
  mpc_.reset();
  mpc_.getSolverPtr()->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
  mpcTimer_.reset();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::setCurrentObservation(const SystemObservation& currentObservation) {
  std::lock_guard<std::mutex> lock(observationMutex_);
  currentObservation_ = currentObservation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ReferenceManagerInterface& MPC_MRT_Interface::getReferenceManager() {
  return mpc_.getSolverPtr()->getReferenceManager();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const ReferenceManagerInterface& MPC_MRT_Interface::getReferenceManager() const {
  return mpc_.getSolverPtr()->getReferenceManager();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::advanceMpc() {
  // measure the delay in running MPC
  mpcTimer_.startTimer();

  SystemObservation currentObservation;
  {
    std::lock_guard<std::mutex> lock(observationMutex_);  
    currentObservation = currentObservation_;
  }
#define USE_CACHING
#ifdef USE_CACHING 
  {
  vector_t feat(12 +4);
  feat << currentObservation.state.head(12), cmdVel;
  
  auto &cache = *m_ocs_caches[currentObservation.mode];
  auto key = cache.quantizeKey(feat);

  double delta;
  if(currentObservation.mode==0 || currentObservation.mode==15) delta = 0.0001;
  else delta=0.1;

  auto solptr=cache.queryNearest(key, feat, delta);
  if ( solptr== nullptr){
    bool controllerIsUpdated = mpc_.run(currentObservation.time, currentObservation.state);
    if (!controllerIsUpdated) {
      return;
    }
    copyToCache(cache, currentObservation, feat, key); 
  }
  else{
  auto& PrimalSolutionptr = solptr->ptr.primalSolution;
  if(controllerPtr_!=nullptr)
    PrimalSolutionptr.controllerPtr_=std::unique_ptr<ocs2::ControllerBase>(controllerPtr_->clone());
  else
    std::cerr << "null pointer found, get good at exception handling" << "\n";

  auto& PerformanceIndexptr = solptr->ptr.performanceIndices;

  auto& CommandDataptr = solptr->ptr.command;
  this->moveToBuffer(std::move(std::make_unique<ocs2::CommandData>(CommandDataptr)),
                     std::move(std::make_unique<ocs2::PrimalSolution>(PrimalSolutionptr)),
                      std::move(std::make_unique<ocs2::PerformanceIndex>(PerformanceIndexptr)));

  }
  } 
  #endif

#ifndef USE_CACHING 
  {

  bool controllerIsUpdated = mpc_.run(currentObservation.time, currentObservation.state);
  if (!controllerIsUpdated) {
      return;
   }
 
  copyToBuffer(currentObservation);
  }
  #endif
  // measure the delay for sending ROS messages
  mpcTimer_.endTimer();

  // check MPC delay and solution window compatibility
  scalar_t timeWindow = mpc_.settings().solutionTimeWindow_;
  if (mpc_.settings().solutionTimeWindow_ < 0) {
    timeWindow = mpc_.getSolverPtr()->getFinalTime() - currentObservation.time;
  }
  if (timeWindow < 2.0 * mpcTimer_.getAverageInMilliseconds() * 1e-3) {
    std::cerr << "[MPC_MRT_Interface::advanceMpc] WARNING: The solution time window might be shorter than the MPC delay!\n";
  }

  // measure the delay
  if (mpc_.settings().debugPrint_) {
    std::cerr << "\n### MPC_MRT Benchmarking";
    std::cerr << "\n###   Maximum : " << mpcTimer_.getMaxIntervalInMilliseconds() << "[ms].";
    std::cerr << "\n###   Average : " << mpcTimer_.getAverageInMilliseconds() << "[ms].";
    std::cerr << "\n###   Latest  : " << mpcTimer_.getLastIntervalInMilliseconds() << "[ms]." << std::endl;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MPC_MRT_Interface::copyToCache(MPCCache_ocs& cache, const SystemObservation& mpcInitObservation, vector_t feat, std::string& key) {
  // policy
  auto primalSolutionPtr = std::make_unique<PrimalSolution>();
  const scalar_t startTime = mpcInitObservation.time;
  const scalar_t finalTime =
      (mpc_.settings().solutionTimeWindow_ < 0) ? mpc_.getSolverPtr()->getFinalTime() : startTime + mpc_.settings().solutionTimeWindow_;
  mpc_.getSolverPtr()->getPrimalSolution(finalTime, primalSolutionPtr.get());

  // command
  auto commandPtr = std::make_unique<CommandData>();
  commandPtr->mpcInitObservation_ = mpcInitObservation;
  commandPtr->mpcTargetTrajectories_ = mpc_.getSolverPtr()->getReferenceManager().getTargetTrajectories();

  // performance indices
  auto performanceIndicesPtr = std::make_unique<PerformanceIndex>();
  *performanceIndicesPtr = mpc_.getSolverPtr()->getPerformanceIndeces();

  controllerPtr_ = (primalSolutionPtr->controllerPtr_)->clone();

  MPCCacheEntry_ocs Entry;
  Entry.feat=feat;

  //commanddata
  Entry.ptr.command.mpcInitObservation_=commandPtr->mpcInitObservation_;
  Entry.ptr.command.mpcTargetTrajectories_=commandPtr->mpcTargetTrajectories_;

  //PrimalSolution
  Entry.ptr.primalSolution = *primalSolutionPtr;

  //Performance index, maybe can remove them?
  Entry.ptr.performanceIndices=*performanceIndicesPtr;


  cache.insert(key, Entry);
  this->moveToBuffer(std::move(commandPtr), std::move(primalSolutionPtr), std::move(performanceIndicesPtr));
}

void MPC_MRT_Interface::copyToBuffer(const SystemObservation& mpcInitObservation) {
  // policy
  auto primalSolutionPtr = std::make_unique<PrimalSolution>();
  const scalar_t startTime = mpcInitObservation.time;
  const scalar_t finalTime =
      (mpc_.settings().solutionTimeWindow_ < 0) ? mpc_.getSolverPtr()->getFinalTime() : startTime + mpc_.settings().solutionTimeWindow_;
  mpc_.getSolverPtr()->getPrimalSolution(finalTime, primalSolutionPtr.get());

  // command
  auto commandPtr = std::make_unique<CommandData>();
  commandPtr->mpcInitObservation_ = mpcInitObservation;
  commandPtr->mpcTargetTrajectories_ = mpc_.getSolverPtr()->getReferenceManager().getTargetTrajectories();

  // performance indices
  auto performanceIndicesPtr = std::make_unique<PerformanceIndex>();
  *performanceIndicesPtr = mpc_.getSolverPtr()->getPerformanceIndeces();

  this->moveToBuffer(std::move(commandPtr), std::move(primalSolutionPtr), std::move(performanceIndicesPtr));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
matrix_t MPC_MRT_Interface::getLinearFeedbackGain(scalar_t time) {
  auto controller = dynamic_cast<LinearController*>(this->getPolicy().controllerPtr_.get());
  if (controller == nullptr) {
    throw std::runtime_error("[MPC_MRT_Interface::getLinearFeedbackGain] Feedback gains only available with linear controller!");
  }
  matrix_t K;
  controller->getFeedbackGain(time, K);
  return K;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation MPC_MRT_Interface::getValueFunction(scalar_t time, const vector_t& state) const {
  return mpc_.getSolverPtr()->getValueFunction(time, state);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t MPC_MRT_Interface::getStateInputEqualityConstraintLagrangian(scalar_t time, const vector_t& state) const {
  return mpc_.getSolverPtr()->getStateInputEqualityConstraintLagrangian(time, state);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MultiplierCollection MPC_MRT_Interface::getIntermediateDualSolution(scalar_t time) const {
  return mpc_.getSolverPtr()->getIntermediateDualSolution(time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
}  // namespace ocs2
