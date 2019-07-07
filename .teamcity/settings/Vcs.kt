package athena.settings

import jetbrains.buildServer.configs.kotlin.v2018_2.vcs.GitVcsRoot

object AthenaAlex : GitVcsRoot({
    name = "athena_alex"
    url = "https://github.com/alexbatashev/athena"
    branch = "refs/heads/develop"
    authMethod = password {
        userName = "alexbatashev"
        password = "zxx5e5775d46ed7c8556a3f134a9241268dc7430c2b41e16c8755c4d1e1696fb2b54131b2cc0bc2468c775d03cbe80d301b"
    }
})

object AthenaAndrey : GitVcsRoot({
    name = "athena_andrey"
    url = "https://github.com/masdevas/athena"
    branch = "refs/heads/develop"
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
