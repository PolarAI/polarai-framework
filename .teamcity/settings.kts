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

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2019.1"

project {

    vcsRoot(HttpsGithubComAthenamlAthenamlGithubIoRefsHeadsMaster)
    vcsRoot(AthenaAlex)
    vcsRoot(AthenaPublic)
    vcsRoot(HttpsGithubComAlexbatashevAthenaRefsHeadsMaster)
    vcsRoot(HttpsGitlabAthenaframeworkMlAthenamlAthenaGitRefsHeadsMaster)
    vcsRoot(GithubGenericRepo)
    vcsRoot(HttpsGithubComAthenamlAthenaRefsHeadsMaster)

    buildType(LinuxGcc8Release)
    buildType(LinuxGcc8Debug)
    buildType(Daily)
    buildType(AlexFork)
    buildType(MandatoryChecks)
    buildType(StaticChecks)
    buildType(LinuxClang8Release)
    buildType(LinuxClang8Debug)
    buildType(UpdateDocs)

    template(Build)

    features {
        feature {
            id = "PROJECT_EXT_10"
            type = "JetBrains.SharedResources"
            param("quota", "1")
            param("name", "ARMVM")
            param("type", "quoted")
            param("enabled", "false")
        }
        feature {
            id = "PROJECT_EXT_11"
            type = "active_storage"
            param("active.storage.feature.id", "DefaultStorage")
        }
        feature {
            id = "PROJECT_EXT_15"
            type = "OAuthProvider"
            param("clientId", "358302cb895dfde1a3e8")
            param("defaultTokenScope", "public_repo,repo,repo:status,write:repo_hook")
            param("secure:clientSecret", "zxxf3671e4259734f488a8f1f91d8699587bb6ee3dfdc7e9b8ceb90ae4ec2aa6230abe6753900b2ba4f775d03cbe80d301b")
            param("displayName", "GitHub.com")
            param("gitHubUrl", "https://github.com/")
            param("providerType", "GitHub")
        }
        youtrack {
            id = "PROJECT_EXT_2"
            displayName = "YouTrack"
            host = "https://youtrack.getathena.ml"
            userName = ""
            password = ""
            projectExtIds = "athena"
            accessToken = "zxx1333982c092e489ffe59a53bc2407b3915d355dffe56a396121652b257197741a1421a4daaa4fb92b8e202cb168c199f4cd4d5132b6840d9c328b1b5f1d960a3"
            param("authType", "accesstoken")
        }
        dockerRegistry {
            id = "PROJECT_EXT_3"
            name = "Gitlab"
            url = "https://registry.gitlab.com"
            userName = "alexbatashev"
            password = "zxx62b4c3dfedc13d8e8efabec4f8aff56e4866455e8f67ead3"
        }
        feature {
            id = "PROJECT_EXT_5"
            type = "IssueTracker"
            param("secure:password", "")
            param("name", "athenaml/athena")
            param("pattern", """#(\d+)""")
            param("authType", "anonymous")
            param("repository", "https://github.com/athenaml/athena")
            param("type", "GithubIssues")
            param("secure:accessToken", "")
            param("username", "")
        }
    }
}

object AlexFork : BuildType({
    name = "Alex Fork"

    type = BuildTypeSettings.Type.COMPOSITE

    params {
        param("reverse.dep.*.repo", "alexbatashev/athena")
        param("reverse.dep.Athena_StaticChecks.target_branch", "develop")
    }

    vcs {
        root(AthenaAlex)

        showDependenciesChanges = true
    }

    triggers {
        vcs {
            quietPeriodMode = VcsTrigger.QuietPeriodMode.USE_DEFAULT
            triggerRules = "+:root=${AthenaAlex.id}:**"

            branchFilter = """
                +:*
                -:develop
                -:master
                -:refs/pull*
            """.trimIndent()
            watchChangesInDependencies = true
            enableQueueOptimization = false
        }
    }

    features {
        commitStatusPublisher {
            vcsRootExtId = "${AthenaAlex.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "zxx1bfacd1e0d69bcbff375fbbc5dfdcebcf0b349a3e1c4d89ea8a537816b12b1fd074ca14f942bc176775d03cbe80d301b"
                }
            }
        }
    }

    dependencies {
        snapshot(LinuxClang8Debug) {
        }
        snapshot(LinuxClang8Release) {
        }
        snapshot(LinuxGcc8Debug) {
        }
        snapshot(LinuxGcc8Release) {
        }
        snapshot(StaticChecks) {
        }
    }
})

object Daily : BuildType({
    name = "Daily"

    type = BuildTypeSettings.Type.COMPOSITE

    vcs {
        root(HttpsGithubComAthenamlAthenaRefsHeadsMaster)

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
        vcsLabeling {
            vcsRootId = "${HttpsGitlabAthenaframeworkMlAthenamlAthenaGitRefsHeadsMaster.id}"
            labelingPattern = "daily-%system.build.number%"
        }
    }

    dependencies {
        snapshot(LinuxClang8Debug) {
        }
        snapshot(LinuxClang8Release) {
        }
        snapshot(LinuxGcc8Debug) {
        }
        snapshot(LinuxGcc8Release) {
        }
        snapshot(StaticChecks) {
        }
    }
})

object LinuxClang8Debug : BuildType({
    templates(Build)
    name = "[Linux][Clang8] Debug"

    artifactRules = "+:%env.ATHENA_BINARY_DIR%/*.html => coverage.zip"

    params {
        param("cc", "clang-8")
        param("image_name", "llvm8")
        param("cxx", "clang++-8")
        param("build_type", "Debug")
        param("build_options", "--ninja --use-ldd --enable-coverage")
    }

    steps {
        script {
            name = "Code coverage"
            id = "RUNNER_29"
            scriptContent = """
                cd %env.ATHENA_BINARY_DIR%
                /usr/local/bin/gcovr . -r %env.SRC_DIR% -o coverage.html --gcov-executable="llvm-cov-8 gcov" --html --html-details -s --exclude-directories=tests --exclude-directories=third_party
            """.trimIndent()
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
            dockerPull = true
        }
    }
})

object LinuxClang8Release : BuildType({
    templates(Build)
    name = "[Linux][Clang8] Release"

    params {
        param("cc", "clang-8")
        param("image_name", "llvm8")
        param("cxx", "clang++-8")
        param("build_type", "Release")
        param("build_options", "--ninja --use-ldd")
    }
})

object LinuxGcc8Debug : BuildType({
    templates(Build)
    name = "[Linux][GCC8] Debug"

    artifactRules = "+:%env.ATHENA_BINARY_DIR%/*.html => coverage.zip"

    params {
        param("cc", "gcc-8")
        param("image_name", "gcc")
        param("cxx", "g++-8")
        param("build_type", "Debug")
        param("build_options", "--enable-coverage --ninja")
    }

    steps {
        script {
            name = "Coverage"
            id = "RUNNER_22"
            scriptContent = """
                cd %env.ATHENA_BINARY_DIR%
                /usr/local/bin/gcovr . -r %env.SRC_DIR% -o coverage.html --gcov-executable=/usr/bin/gcov-8 --html --html-details --exclude-directories=tests --exclude-directories=third_party --exclude-unreachable-branches -s > covreport.txt
                LCOVERAGE=`grep "lines: *" covreport.txt | sed -e 's/lines: //g' | sed -e 's/%.*//g'`
                BCOVERAGE=`grep "branches: *" covreport.txt | sed -e 's/branches: //g' | sed -e 's/%.*//g'`
                echo "##teamcity[buildStatisticValue key='CodeCoverageL' value='""${'$'}LCOVERAGE""']"
                echo "##teamcity[buildStatisticValue key='CodeCoverageB' value='""${'$'}BCOVERAGE""']"
            """.trimIndent()
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerImagePlatform = ScriptBuildStep.ImagePlatform.Linux
        }
    }

    failureConditions {
        failOnMetricChange {
            id = "BUILD_EXT_12"
            metric = BuildFailureOnMetric.MetricType.COVERAGE_LINE_PERCENTAGE
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.LESS
            compareTo = build {
                buildRule = lastSuccessful()
            }
            param("buildTag", "nightly")
        }
        failOnMetricChange {
            id = "BUILD_EXT_13"
            metric = BuildFailureOnMetric.MetricType.COVERAGE_LINE_PERCENTAGE
            threshold = 80
            units = BuildFailureOnMetric.MetricUnit.DEFAULT_UNIT
            comparison = BuildFailureOnMetric.MetricComparison.LESS
            compareTo = value()
            param("anchorBuild", "lastSuccessful")
        }
    }
})

object LinuxGcc8Release : BuildType({
    templates(Build)
    name = "[Linux][GCC8]Release"

    params {
        param("cc", "gcc-8")
        param("image_name", "gcc")
        param("cxx", "g++-8")
        param("build_type", "Release")
        param("build_options", "--ninja")
    }
})

object MandatoryChecks : BuildType({
    name = "Pull Request / Post commit"

    type = BuildTypeSettings.Type.COMPOSITE

    params {
        param("reverse.dep.*.repo", "athenaml/athena")
        param("reverse.dep.Athena_StaticChecks.target_branch", "develop")
    }

    vcs {
        root(AthenaPublic)

        showDependenciesChanges = true
    }

    triggers {
        vcs {
            quietPeriodMode = VcsTrigger.QuietPeriodMode.USE_DEFAULT
            triggerRules = "+:root=${AthenaPublic.id}:**"

            branchFilter = """
                +:develop
                +:master
            """.trimIndent()
            watchChangesInDependencies = true
            enableQueueOptimization = false
        }
    }

    features {
        pullRequests {
            vcsRootExtId = "${AthenaPublic.id}"
            provider = github {
                authType = vcsRoot()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
        commitStatusPublisher {
            vcsRootExtId = "${AthenaPublic.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = personalToken {
                    token = "zxx1bfacd1e0d69bcbff375fbbc5dfdcebcf0b349a3e1c4d89ea8a537816b12b1fd074ca14f942bc176775d03cbe80d301b"
                }
            }
        }
        commitStatusPublisher {
            vcsRootExtId = "${AthenaPublic.id}"
            publisher = upsource {
                serverUrl = "https://upsource.getathena.ml"
                projectId = "ATHENA"
                userName = "admin"
                password = "zxx771e317223635f9b139d6aec1e0d771ca7964e61aa6cbd40604de29877cb04b50a8d93941b3b5f7ce52d4fa5a8dfbd56ebe49af0469bf9f4626f1248aebc6d0d775d03cbe80d301b"
            }
        }
    }

    dependencies {
        snapshot(LinuxClang8Debug) {
        }
        snapshot(LinuxClang8Release) {
        }
        snapshot(LinuxGcc8Debug) {
        }
        snapshot(LinuxGcc8Release) {
        }
        snapshot(StaticChecks) {
        }
    }
})

object StaticChecks : BuildType({
    name = "Static Checks"

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
        root(GithubGenericRepo)

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
    type = BuildTypeSettings.Type.DEPLOYMENT
    maxRunningBuilds = 1
    publishArtifacts = PublishMode.ALWAYS

    vcs {
        root(HttpsGithubComAthenamlAthenaRefsHeadsMaster)
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
                +:/refs/heads/master
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
            vcsRootExtId = "${HttpsGithubComAthenamlAthenaRefsHeadsMaster.id}"
            provider = github {
                authType = vcsRoot()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
            }
        }
        pullRequests {
            vcsRootExtId = "${HttpsGithubComAlexbatashevAthenaRefsHeadsMaster.id}"
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

object Build : Template({
    name = "Build"

    params {
        param("cc", "")
        param("image_name", "")
        param("cxx", "")
        param("env.BUILD_TYPE", "%build_type%")
        param("repo", "athenaml/athena")
        param("env.SRC_DIR", "%system.teamcity.build.checkoutDir%")
        param("env.ATHENA_BINARY_DIR", "%teamcity.build.checkoutDir%/build")
        param("build_type", "")
    }

    vcs {
        root(GithubGenericRepo)
    }

    steps {
        exec {
            name = "Build"
            id = "RUNNER_5"
            path = "scripts/build.py"
            arguments = "%build_options% --build-type %build_type% %env.ATHENA_BINARY_DIR% %env.SRC_DIR%"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
            dockerRunParameters = "-e CC=%cc% -e CXX=%cxx% -e ATHENA_BINARY_DIR=%env.ATHENA_BINARY_DIR% -e ATHENA_TEST_ENVIRONMENT=ci"
        }
        exec {
            name = "Test"
            id = "RUNNER_6"
            path = "scripts/test.sh"
            dockerImage = "registry.gitlab.com/athenaml/ubuntu_docker_images/%image_name%:latest"
            dockerPull = true
        }
    }

    features {
        dockerSupport {
            id = "DockerSupport"
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_3"
            }
        }
        feature {
            id = "BUILD_EXT_7"
            type = "xml-report-plugin"
            param("xmlReportParsing.reportType", "ctest")
            param("xmlReportParsing.reportDirs", "+:build/Testing/**/*.xml")
        }
    }

    requirements {
        equals("docker.server.osType", "linux", "RQ_1")
    }
})

object AthenaAlex : GitVcsRoot({
    name = "athena_alex"
    url = "https://github.com/alexbatashev/athena"
    branch = "refs/heads/develop"
    authMethod = password {
        userName = "alexbatashev"
        password = "zxx5e5775d46ed7c8556a3f134a9241268dc7430c2b41e16c8755c4d1e1696fb2b54131b2cc0bc2468c775d03cbe80d301b"
    }
})

object AthenaPublic : GitVcsRoot({
    name = "athena_public"
    url = "https://github.com/athenaml/athena"
    branch = "refs/heads/develop"
    authMethod = password {
        userName = "alexbatashev"
        password = "zxx5e5775d46ed7c8556a3f134a9241268dc7430c2b41e16c8755c4d1e1696fb2b54131b2cc0bc2468c775d03cbe80d301b"
    }
})

object GithubGenericRepo : GitVcsRoot({
    name = "Github Generic Repo"
    url = "https://github.com/%repo%"
    branch = "refs/heads/develop"
})

object HttpsGithubComAlexbatashevAthenaRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/alexbatashev/athena#refs/heads/master"
    url = "https://github.com/alexbatashev/athena"
    authMethod = password {
        userName = "alexbatashev"
        password = "zxx5e5775d46ed7c8556a3f134a9241268dc7430c2b41e16c8755c4d1e1696fb2b54131b2cc0bc2468c775d03cbe80d301b"
    }
})

object HttpsGithubComAthenamlAthenaRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/athena#refs/heads/master"
    url = "https://github.com/athenaml/athena"
    authMethod = password {
        userName = "alexbatashev"
        password = "zxx5e5775d46ed7c8556a3f134a9241268dc7430c2b41e16c8755c4d1e1696fb2b54131b2cc0bc2468c775d03cbe80d301b"
    }
})

object HttpsGithubComAthenamlAthenamlGithubIoRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/athenaml.github.io#refs/heads/master"
    url = "https://github.com/athenaml/athenaml.github.io"
    authMethod = password {
        userName = "alexbatashev"
        password = "zxx766def144dcfe0adaeb9deff879a973c32b50763bf51b6653295b24a8370526a509cc1c746f8c478775d03cbe80d301b"
    }
})

object HttpsGithubComAthenamlWebsiteRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/website#refs/heads/master"
    url = "https://github.com/athenaml/website"
    authMethod = password {
        userName = "alexbatashev"
        password = "zxx766def144dcfe0adaeb9deff879a973c32b50763bf51b6653295b24a8370526a509cc1c746f8c478775d03cbe80d301b"
    }
})
