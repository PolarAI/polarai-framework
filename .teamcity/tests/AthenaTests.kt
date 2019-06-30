import org.junit.Assert.assertTrue
import org.junit.Test

class AthenaTests {

    @Test
    fun buildsHaveCleanCheckOut() {
        val project = AthenaProject

        project.buildTypes.forEach { bt ->
            assertTrue("BuildType '${bt.id}' doesn't use clean checkout",
                    bt.vcs.cleanCheckout)
        }
    }
}