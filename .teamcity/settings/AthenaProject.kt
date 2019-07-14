package athena.settings

import jetbrains.buildServer.configs.kotlin.v2018_2.Project
import jetbrains.buildServer.configs.kotlin.v2018_2.projectFeatures.dockerRegistry
import jetbrains.buildServer.configs.kotlin.v2018_2.projectFeatures.youtrack

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

val compilers = listOf("Clang", "GCC")
val buildConfigs = listOf("Release", "Debug")
val repos = listOf(AthenaPublic, AthenaAlex, AthenaAndrey)

object AthenaProject : Project({

    subProject {
        id("AthenaPublicRepo")
        name = "Athena Public"
        vcsRoot(AthenaPublic)
        vcsRoot(HttpsGithubComAthenamlWebsiteRefsHeadsMaster)
        vcsRoot(HttpsGithubComAthenamlAthenamlGithubIoRefsHeadsMaster)
        for (compiler in compilers) {
            for (buildConfig in buildConfigs) {
                buildType(DefaultBuild(AthenaPublic, buildConfig, compiler))
            }
        }
        buildType(Daily)
        buildType(StaticChecks(AthenaPublic))
        buildType(MandatoryChecks(AthenaPublic))
        buildType(UpdateDocs)
    }

    subProject {
        id("AthenaAlexFork")
        name = "Alex Fork"
        vcsRoot(AthenaAlex)
        for (compiler in compilers) {
            for (buildConfig in buildConfigs) {
                buildType(DefaultBuild(AthenaAlex, buildConfig, compiler))
            }
        }
        buildType(StaticChecks(AthenaAlex))
        buildType(MandatoryChecks(AthenaAlex))
    }

    subProject {
        id("AthenaAndreyFork")
        name = "Andrey Fork"
        vcsRoot(AthenaAndrey)
        for (compiler in compilers) {
            for (buildConfig in buildConfigs) {
                buildType(DefaultBuild(AthenaAndrey, buildConfig, compiler))
            }
        }
        buildType(StaticChecks(AthenaAndrey))
        buildType(MandatoryChecks(AthenaAndrey))
    }

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
            userName = "athenamlbot"
            password = "zxx6ed3175f54e3c3f10f176feb94d281ab48c3a1fd7c269d45"
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
})