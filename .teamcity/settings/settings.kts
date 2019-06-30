import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.dockerSupport
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.investigationsAutoAssigner
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.sshAgent
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.vcsLabeling
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.ExecBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.exec
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.script
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.failOnText
import jetbrains.buildServer.configs.kotlin.v2018_2.projectFeatures.dockerRegistry
import jetbrains.buildServer.configs.kotlin.v2018_2.projectFeatures.youtrack
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.VcsTrigger
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.schedule
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs
import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

version = "2019.1"

project(AthenaProject)