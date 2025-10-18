# GitHub Actions CI/CD Template for CUDA Projects

This directory contains CI/CD workflow templates for professional CUDA C++ development.

## Available Workflows

### ci-cuda-full.yml (Recommended)

A comprehensive CI/CD pipeline that includes:

- **Code Quality Checks**:

  - Pre-commit hooks validation
  - Code formatting verification
  - Style compliance

- **Build and Test**:

  - Multi-platform build support
  - Conan dependency caching
  - Automated testing with CTest
  - Test result artifacts

- **Static Analysis**:

  - clang-tidy analysis
  - Compile commands export
  - Analysis result artifacts

- **Documentation**:

  - Doxygen documentation generation
  - Automatic GitHub Pages deployment
  - Documentation artifacts

- **CUDA Testing** (Optional):

  - GPU runner support
  - Memory leak detection
  - CUDA-specific checks

## Setup Instructions

### 1. Enable GitHub Actions

Copy the template workflow to your repository:

```bash
# Use the comprehensive workflow
cp .github/workflows/ci-cuda-full.yml.template .github/workflows/ci.yml

# Or create a minimal workflow
cp .github/workflows/ci-minimal.yml.template .github/workflows/ci.yml
```

### 2. Configure Repository Settings

1. Go to repository Settings → Actions → General
1. Enable "Read and write permissions" for workflows
1. Allow GitHub Actions to create pull requests (if needed)

### 3. Enable GitHub Pages (for documentation)

1. Go to Settings → Pages
1. Source: "GitHub Actions"
1. Documentation will be published automatically on main branch pushes

### 4. Add Self-hosted GPU Runner (Optional)

For CUDA testing with GPU access:

1. Go to Settings → Actions → Runners
1. Click "New self-hosted runner"
1. Follow instructions to set up a machine with NVIDIA GPU
1. Tag the runner as `gpu`
1. Uncomment the `cuda-test` job in the workflow

### 5. Configure Branch Protection

Recommended branch protection rules for `main`:

1. Settings → Branches → Add rule
1. Branch name pattern: `main`
1. Enable:
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - Select required checks:
     - `Code Quality Checks`
     - `Build and Test`
     - `Static Analysis`
     - `Build Documentation`
   - ✅ Require linear history (optional)
   - ✅ Include administrators

## Workflow Triggers

All workflows are triggered by:

- **Push** to `main` or `develop` branches
- **Pull requests** to `main` or `develop` branches

You can customize triggers in the workflow file:

```yaml
on:
  push:
    branches: [ main, develop, feature/* ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:  # Manual trigger
```

## Caching Strategy

The workflows use GitHub Actions cache to speed up builds:

### Conan Cache

```yaml
- uses: actions/cache@v3
  with:
    path: ~/.conan2
    key: ${{ runner.os }}-conan-${{ hashFiles('**/conanfile.py') }}
```

### Build Cache (optional)

```yaml
- uses: actions/cache@v3
  with:
    path: build
    key: ${{ runner.os }}-build-${{ hashFiles('**/*.cpp', '**/*.cu') }}
```

## Artifacts

Workflows produce the following artifacts:

| Artifact                  | Description                    | Retention |
| ------------------------- | ------------------------------ | --------- |
| `test-results`            | CTest output and logs          | 30 days   |
| `static-analysis-results` | clang-tidy output              | 30 days   |
| `documentation`           | Generated HTML docs            | 90 days   |
| `memory-check-results`    | Valgrind/cuda-memcheck reports | 30 days   |

Access artifacts from the Actions tab → Workflow run → Artifacts section.

## Status Badges

Add status badges to your README.md:

```markdown
![CI](https://github.com/USERNAME/REPO/workflows/CUDA%20C++%20CI/badge.svg)
![Documentation](https://github.com/USERNAME/REPO/workflows/Documentation/badge.svg)
```

## Customization

### Adding New Jobs

```yaml
my-custom-job:
  name: My Custom Job
  runs-on: ubuntu-latest
  needs: [build-and-test]  # Run after build-and-test

  steps:
  - uses: actions/checkout@v3
  - name: Run custom step
    run: echo "Hello, World!"
```

### Matrix Builds

Test multiple configurations:

```yaml
build-and-test:
  strategy:
    matrix:
      os: [ubuntu-latest, ubuntu-20.04]
      build_type: [Release, Debug]
      cuda_version: ['11.8', '12.0']

  runs-on: ${{ matrix.os }}

  steps:
  - name: Build
    run: cmake -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} ...
```

### Secrets

For private dependencies or deployment:

1. Settings → Secrets and variables → Actions
1. Add repository secrets
1. Use in workflow: `${{ secrets.MY_SECRET }}`

## Troubleshooting

### Build Fails

- Check artifact logs in Actions tab
- Verify dependencies in `conanfile.py`
- Test locally: `make build`

### Tests Fail

- Review test results artifact
- Run locally: `make test`
- Check CTest output: `cd build/build/Release && ctest -V`

### Documentation Fails

- Check Doxygen configuration in `Doxyfile`
- Test locally: `make docs`
- Verify Doxygen comments syntax

### Cache Issues

- Clear cache: Settings → Actions → Caches
- Or add `-force-cache-update` to workflow

## Best Practices

1. **Run checks locally first**: Use `make format`, `make lint`, `make test`
1. **Use pre-commit hooks**: Catch issues before pushing
1. **Keep workflows fast**: Use caching and parallel jobs
1. **Monitor workflow usage**: GitHub has usage limits
1. **Update actions regularly**: Keep action versions current
1. **Document custom workflows**: Add comments in workflow files

## Example Workflow Files

### Minimal CI (ci-minimal.yml)

```yaml
name: Minimal CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: make build
      - name: Test
        run: make test
```

### Nightly Build (nightly.yml)

```yaml
name: Nightly Build

on:
  schedule:
    - cron: '0 0 * * *'  # Every day at midnight

jobs:
  comprehensive-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Full test suite
        run: |
          make build
          make test
          make memory-check
          make lint
```

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [GitHub Actions Cache](https://docs.github.com/en/actions/guides/caching-dependencies-to-speed-up-workflows)
- [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)

## Support

For issues with workflows:

1. Check Actions tab for detailed logs
1. Review [GitHub Actions documentation](https://docs.github.com/actions)
1. Open an issue with workflow logs attached
