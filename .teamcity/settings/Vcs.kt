package athena.settings

import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

object AthenaAlex : GitVcsRoot({
    name = "athena_alex"
    url = "https://github.com/alexbatashev/athena"
    branch = "refs/heads/develop"
    branchSpec = """
        +:refs/heads/master
        +:refs/pull/*
        +:refs/pull/*/merge
    """.trimIndent()
    authMethod = password {
        userName = "athenamlbot"
        password = "zxx24d3eb84f48caad36dc56a3252875b70b7fd09cd46cb8ab54153e54fc1042983f410212252a6a0cf775d03cbe80d301b"
    }
})

object AthenaAndrey : GitVcsRoot({
    name = "athena_andrey"
    url = "https://github.com/masdevas/athena"
    branch = "refs/heads/develop"
    branchSpec = """
        +:refs/heads/master
        +:refs/pull/*
        +:refs/pull/*/merge
    """.trimIndent()
    authMethod = password {
        userName = "athenamlbot"
        password = "zxx24d3eb84f48caad36dc56a3252875b70b7fd09cd46cb8ab54153e54fc1042983f410212252a6a0cf775d03cbe80d301b"
    }
})

object AthenaPublic : GitVcsRoot({
    name = "athena_public"
    url = "https://github.com/athenaml/athena"
    branch = "refs/heads/develop"
    branchSpec = """
        +:refs/heads/master
        +:refs/pull/*
        +:refs/pull/*/merge
    """.trimIndent()
    authMethod = password {
        userName = "athenamlbot"
        password = "zxx24d3eb84f48caad36dc56a3252875b70b7fd09cd46cb8ab54153e54fc1042983f410212252a6a0cf775d03cbe80d301b"
    }
})

object HttpsGithubComAthenamlAthenamlGithubIoRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/athenaml.github.io#refs/heads/master"
    url = "https://github.com/athenaml/athenaml.github.io"
    authMethod = password {
        userName = "athenamlbot"
        password = "zxx24d3eb84f48caad36dc56a3252875b70b7fd09cd46cb8ab54153e54fc1042983f410212252a6a0cf775d03cbe80d301b"
    }
})

object HttpsGithubComAthenamlWebsiteRefsHeadsMaster : GitVcsRoot({
    name = "https://github.com/athenaml/website#refs/heads/master"
    url = "https://github.com/athenaml/website"
    authMethod = password {
        userName = "athenamlbot"
        password = "zxx24d3eb84f48caad36dc56a3252875b70b7fd09cd46cb8ab54153e54fc1042983f410212252a6a0cf775d03cbe80d301b"
    }
})
