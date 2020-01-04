# How to contribute

Everyone is welcome to contribute to Athena project. There are several ways to get involved.

## Community Guidelines

Please, get familiar with our [Code of conduct](CODE_OF_CONDUCT.md) which aims
to foster an open and welcoming environment.

## Issue tracking and reporting

To report a bug, please use [Github Issues](https://github.com/athenaml/athena/issues).

If you want to contribute to the project, start looking for issues with `good first issue`
label. If you decide to start working on issue, leave a comment, so that other
community members are aware of work in progress.

## Contributing code

### Code style

Athena mostly follows the [LLVM style](https://llvm.org/docs/CodingStandards.html) guide with
a few exceptions:

* Adopts [camelBack](https://llvm.org/docs/Proposals/VariableNames.html).
* `*` and `&` are "attached" to the type (e.g `int* a`).
* Outside the LLVM backend standard types, algorithms and containers are used. (Do not re-invent the wheel).

There's a `.clang-format` configuration file to help you properly format your sources.
Please, always apply clang-format to *your patch* (do not reformat code outside scope of
your task).

**Note**

As of moment of writing these lines, not all sources follow aforementioned conventions.
For example, most of doxygen comments use `/** */` style comments. Improving consistency 
of code style is an ongoing process. It might take a while. If you find yourself uncomfortable
following some of LLVM guidelines, match your code style to the style used in current file.

### Commit messages

Athena follows [Conventional commits](https://www.conventionalcommits.org/) guidelines.

#### Commit message format

Each commit message consists of a **header**, a **body** and a **footer**.  The header has a special
format that includes a **type**, a **scope** and a **subject**:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **header** is mandatory and the **scope** of the header is optional.

Any line of the commit message cannot be longer 100 characters! This allows the 
message to be easier to read on GitHub as well as in various git tools.

The footer should contain a [closing reference to an issue](https://help.github.com/articles/closing-issues-via-commit-messages/) if any.

#### Revert

If the commit reverts a previous commit, it should begin with `revert: `, followed by the header of the 
reverted commit. In the body it should say: `This reverts commit <hash>.`, where the hash is the SHA of the 
commit being reverted.

#### Type

Must be one of the following:

* **build**: Changes that affect the build system or external dependencies
* **ci**: Changes to our CI configuration files and scripts
* **docs**: Documentation only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* **test**: Adding missing tests or correcting existing tests

#### Scope

The scope should be the name of the component affected. Examples are listed below:

* **core** - Athena core runtime library (`src/core`).
* **ops** - Operations (`src/ops`).
* **loaders** - Loaders (`src/loaders`).
* **llvm_backend** - LLVM backend (`src/backend/llvm`).
* **cpu_rt** - CPU runtime (`src/backend/llvm/runtime/cpu`).
* **rt_driver** - Runtime driver (`src/backend/llvm/runtime/driver`).
* **mlir_graph** - MLIR-based Graph dialect (`src/backend/llvm/mlir`).

#### Subject

The subject contains a succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize the first letter
* no dot (.) at the end

#### Body

Just as in the **subject**, use the imperative, present tense: "change" not "changed" nor "changes".
The body should include the motivation for the change and contrast this with previous behavior.

#### Footer

The footer should contain any information about **Breaking Changes** and is also the place to
reference GitHub issues that this commit **Closes**.

**Breaking Changes** should start with the word `BREAKING CHANGE:` with a space or two newlines. The rest of the commit message is then used for this.
