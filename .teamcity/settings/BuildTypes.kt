package athena.settings

import jetbrains.buildServer.configs.kotlin.v10.toExtId
import jetbrains.buildServer.configs.kotlin.v2018_2.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildFeatures.*
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.ExecBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.ScriptBuildStep
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.exec
import jetbrains.buildServer.configs.kotlin.v2018_2.buildSteps.script
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.BuildFailureOnMetric
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.BuildFailureOnText
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.failOnMetricChange
import jetbrains.buildServer.configs.kotlin.v2018_2.failureConditions.failOnText
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.VcsTrigger
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.schedule
import jetbrains.buildServer.configs.kotlin.v2018_2.triggers.vcs
import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

class DefaultBuild(private val repo: GitVcsRoot, private val buildConfig: String, private val compiler: String) : BuildType({
    id("AthenaBuild_${repo.name}_${buildConfig}_$compiler".toExtId())
    name = "[$buildConfig][$compiler] Build"

    var buildOptions = "--ninja"

    if (compiler == "Clang") {
        buildOptions += " --use-ldd"
    }

    if (buildConfig == "Debug") {
        buildOptions += " --enable-coverage"
    }

    params {
        param("cc", (if (compiler == "Clang") "clang-8" else "gcc"))
        param("image_name", (if (compiler == "Clang") "llvm8" else "gcc"))
        param("cxx", (if (compiler == "Clang") "clang++-8" else "g++"))
        param("env.BUILD_TYPE", "%build_type%")
        param("repo", "athenaml/athena")
        param("env.SRC_DIR", "%system.teamcity.build.checkoutDir%")
        param("env.ATHENA_BINARY_DIR", "%teamcity.build.checkoutDir%/build_${buildConfig}_$compiler")
        param("build_type", "")
        param("build_options", "--ninja")
    }

    steps {
        exec {
            name = "Build"
            path = "scripts/build.py"
            arguments = "$buildOptions --build-type $buildConfig %env.ATHENA_BINARY_DIR% %env.SRC_DIR%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
            dockerRunParameters = "-e CC=%cc% -e CXX=%cxx% -e ATHENA_BINARY_DIR=%env.ATHENA_BINARY_DIR% -e ATHENA_TEST_ENVIRONMENT=ci"
        }
        exec {
            name = "Test"
            path = "scripts/test.sh"
            arguments = "%env.ATHENA_BINARY_DIR%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
        }
    }

    vcs {
        root(repo)
    }

    features {
        dockerSupport {
            id = "DockerSupport"
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
        feature {
            type = "xml-report-plugin"
            param("xmlReportParsing.reportType", "ctest")
            param("xmlReportParsing.reportDirs", "+:build_${buildConfig}_${compiler}/Testing/**/*.xml")
        }
    }

    requirements {
        equals("docker.server.osType", "linux")
    }
})

object Daily : BuildType({
    name = "Daily"

    type = Type.COMPOSITE

    vcs {
        root(AthenaPublic)

        showDependenciesChanges = true
    }

    triggers {
        schedule {
            schedulingPolicy = daily {
                timezone = "Europe/Moscow"
            }
            branchFilter = "+:<default>"
            triggerRules = "+:"
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }

    features {
        investigationsAutoAssigner {
            defaultAssignee = "abatashev"
        }
    }

    dependencies {
        for (compiler in compilers) {
            for (buildConfig in buildConfigs) {
                snapshot(DefaultBuild(AthenaPublic, buildConfig, compiler)) {}
            }
        }
        snapshot(StaticChecks(AthenaPublic)) {}
    }
})

class MandatoryChecks(private val repo: GitVcsRoot) : BuildType({
    name = "Pull Request / Post commit"
    id("AthenaMandatoryChecks_${repo.name}")

    type = Type.COMPOSITE

    vcs {
        root(repo)
        showDependenciesChanges = true
    }

    triggers {
        vcs {
            quietPeriodMode = VcsTrigger.QuietPeriodMode.USE_DEFAULT
            triggerRules = "+:root=${repo.id}:**"

            if (repo.name == "athena_public") {
                branchFilter = """
                +:refs/pull/(*/merge)
                +:develop
                +:master
                """.trimIndent()
            } else {
                branchFilter = "+:refs/heads/*"
            }
            watchChangesInDependencies = true
            enableQueueOptimization = false
        }
    }

    if (repo.name == "athena_public") {
        features {
            pullRequests {
                vcsRootExtId = "${repo.id}"
                provider = github {
                    authType = vcsRoot()
                    filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
                }
            }
            commitStatusPublisher {
                vcsRootExtId = "${repo.id}"
                publisher = github {
                    githubUrl = "https://api.github.com"
                    authType = personalToken {
                        token = "zxx1bfacd1e0d69bcbff375fbbc5dfdcebcf0b349a3e1c4d89ea8a537816b12b1fd074ca14f942bc176775d03cbe80d301b"
                    }
                }
            }
            commitStatusPublisher {
                vcsRootExtId = "${repo.id}"
                publisher = upsource {
                    serverUrl = "https://upsource.getathena.ml"
                    projectId = "ATHENA"
                    userName = "admin"
                    password = "zxx771e317223635f9b139d6aec1e0d771ca7964e61aa6cbd40604de29877cb04b50a8d93941b3b5f7ce52d4fa5a8dfbd56ebe49af0469bf9f4626f1248aebc6d0d775d03cbe80d301b"
                }
            }
        }
    }

    dependencies {
        for (compiler in compilers) {
            for (buildConfig in buildConfigs) {
                snapshot(DefaultBuild(repo, buildConfig, compiler)) {}
            }
        }
        snapshot(StaticChecks(repo)) {}
    }
})

class StaticChecks(private val repo: GitVcsRoot) : BuildType({
    name = "Static Checks"
    id("AthenaStaticChecks_${repo.name}")

    artifactRules = """
        +:format-fixes.diff
        +:tidy-fixes.yaml
        +:infer-out/** => infer.zip
    """.trimIndent()

    params {
        param("env.SRC_DIR", "%system.teamcity.build.checkoutDir%")
        param("target_branch", "")
        param("repo", "athenaml/athena")
    }

    vcs {
        root(repo)

        cleanCheckout = true
    }

    steps {
        exec {
            name = "Get compile commands"
            path = "scripts/build.py"
            arguments = "--no-build --ninja --disable-tests --compile-commands %teamcity.build.checkoutDir%/comp_comm %teamcity.build.checkoutDir%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/llvm8:latest"
            dockerRunParameters = "-e CC=clang-8 -e CXX=clang++-8"
        }
        script {
            name = "Clang Tidy"
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            scriptContent = "run-clang-tidy-8.py -export-fixes=tidy-fixes.yaml -header-filter='%teamcity.build.checkoutDir%/include' -p=%teamcity.build.checkoutDir%/comp_comm %teamcity.build.checkoutDir%/src"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/llvm8:latest"
            dockerPull = true
        }
        script {
            name = "Infer"
            enabled = false
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            scriptContent = "infer --compilation-database %teamcity.build.checkoutDir%/comp_comm/compilation_database.json --html"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/infer:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
        }
        exec {
            name = "Clang Format"
            executionMode = BuildStep.ExecutionMode.RUN_ON_FAILURE
            path = "scripts/clang-format.sh"
            arguments = "%target_branch%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/llvm8:latest"
            dockerImagePlatform = ExecBuildStep.ImagePlatform.Linux
            dockerPull = true
            dockerRunParameters = "-e CLANG_FORMAT_DIFF_EXEC=/usr/lib/llvm-8/share/clang/clang-format-diff.py"
        }
    }

    failureConditions {
        failOnText {
            enabled = false
            conditionType = BuildFailureOnText.ConditionType.CONTAINS
            pattern = "warning:"
            failureMessage = "There were errors on static checks"
            reverse = false
        }
        failOnMetricChange {
            metric = BuildFailureOnMetric.MetricType.ARTIFACT_SIZE
            threshold = 40
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.MORE
            compareTo = value()
            param("anchorBuild", "lastSuccessful")
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
    }

    cleanup {
        artifacts(builds = 1, days = 7)
    }
})

object UpdateDocs : BuildType({
    name = "Update Docs"

    enablePersonalBuilds = false
    artifactRules = """
        +:website/build => docs.zip
        +:website/*.log => logs.zip
    """.trimIndent()
    type = Type.DEPLOYMENT
    maxRunningBuilds = 1
    publishArtifacts = PublishMode.ALWAYS

    vcs {
        root(AthenaPublic)
        root(HttpsGithubComAthenamlWebsiteRefsHeadsMaster, "+:. => website")
    }

    steps {
        script {
            name = "Generate doxygen docs"
            scriptContent = """
                scripts/build.py --no-build --docs-only --build-type Debug %teamcity.build.checkoutDir%/doxygen %teamcity.build.checkoutDir%
                cmake --build %teamcity.build.checkoutDir%/doxygen --target docs
            """.trimIndent()
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/docs:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
        }
        script {
            name = "Build website"
            scriptContent = """
                cd %teamcity.build.checkoutDir%/website
                npm install
                npm run build
                mkdir -p %teamcity.build.checkoutDir%/website/build/athenaml.github.io/api/develop
                cp -R %teamcity.build.checkoutDir%/doxygen/docs/* %teamcity.build.checkoutDir%/website/build/athenaml.github.io/api/develop/
            """.trimIndent()
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/docs:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
        }
        script {
            name = "Deploy"
            scriptContent = """
                cd %teamcity.build.checkoutDir%/website
                git config --global user.email "bot@getathena.ml"
                git config --global user.name "Athena Build Bot"
                git clone git@github.com:athenaml/athenaml.github.io.git
                cp -R build/athenaml.github.io/* athenaml.github.io/
                cd athenaml.github.io
                git add .
                git commit -m "Update website"
                git push --all
                cd ..
                rm -r athenaml.github.io
            """.trimIndent()
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
        }
    }

    triggers {
        vcs {
            branchFilter = """
                +:refs/heads/master
                +:<default>
            """.trimIndent()
        }
    }

    features {
        dockerSupport {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
        sshAgent {
            teamcitySshKey = "teamcity_github"
        }
        pullRequests {
            vcsRootExtId = "${AthenaPublic.id}"
            provider = github {
                authType = vcsRoot()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
        pullRequests {
            vcsRootExtId = "${AthenaAlex.id}"
            provider = github {
                authType = vcsRoot()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
    }

    requirements {
        equals("system.agent.name", "Default Agent")
    }
})