ENV ?= land-research
RENDERED := ../.lithops/$(ENV)
REPO_ROOT := ..

.PHONY: help render clean gate-a docker-gitsha \
        build-loaders-usw2 build-loaders-euc1 build-s1 build-predict build-all \
        deploy-predict

help:
	@echo "Targets (pass ENV=<env>, default: $(ENV)):"
	@echo "  render               Render Lithops configs from Terraform outputs"
	@echo "  gate-a               Build predict image locally + verify vs golden tile"
	@echo "  build-loaders-usw2   lithops runtime build ttc-loaders-dev (us-west-2)"
	@echo "  build-loaders-euc1   lithops runtime build ttc-loaders-dev (eu-central-1)"
	@echo "  build-s1             lithops runtime build ttc-s1-dev"
	@echo "  build-predict        lithops runtime build ttc-predict-dev"
	@echo "  build-all            All four runtimes"
	@echo "  deploy-predict       Eagerly deploy predict Lambda (optional; Lithops deploys lazily)"
	@echo "  clean                Remove rendered .lithops/\$$(ENV)"

render:
	uv run python lithops/render.py --env $(ENV)

$(RENDERED)/config.loaders-usw2.yaml \
$(RENDERED)/config.loaders-euc1.yaml \
$(RENDERED)/config.s1.yaml \
$(RENDERED)/config.predict.yaml: render

build-loaders-usw2: $(RENDERED)/config.loaders-usw2.yaml
	cd $(REPO_ROOT) && uv run lithops runtime build \
	    -f docker/PipDockerfile -b aws_lambda \
	    -c .lithops/$(ENV)/config.loaders-usw2.yaml ttc-loaders-dev

build-loaders-euc1: $(RENDERED)/config.loaders-euc1.yaml
	cd $(REPO_ROOT) && uv run lithops runtime build \
	    -f docker/PipDockerfile -b aws_lambda \
	    -c .lithops/$(ENV)/config.loaders-euc1.yaml ttc-loaders-dev

build-s1: $(RENDERED)/config.s1.yaml
	cd $(REPO_ROOT) && uv run lithops runtime build \
	    -f docker/PipDockerfile -b aws_lambda \
	    -c .lithops/$(ENV)/config.s1.yaml ttc-s1-dev

# Bake the pipeline's current git sha into docker/.git_sha so the predict
# image can read it at runtime (env-var injection isn't available through
# `lithops runtime build`, so we ship the sha as a file instead).
docker-gitsha:
	@git -C $(REPO_ROOT) rev-parse HEAD > $(REPO_ROOT)/docker/.git_sha 2>/dev/null \
	    || echo unknown > $(REPO_ROOT)/docker/.git_sha
	@echo "docker/.git_sha := $$(cat $(REPO_ROOT)/docker/.git_sha)"

build-predict: $(RENDERED)/config.predict.yaml docker-gitsha
	cd $(REPO_ROOT) && uv run lithops runtime build \
	    -f docker/PredictDockerfile -b aws_lambda \
	    -c .lithops/$(ENV)/config.predict.yaml ttc-predict-dev

build-all: build-loaders-usw2 build-loaders-euc1 build-s1 build-predict

# Eager Lambda deploy. Useful to surface config/role errors before a pipeline run.
# Lithops will otherwise create the function on first invocation.
deploy-predict: build-predict
	cd $(REPO_ROOT) && uv run lithops runtime deploy \
	    -b aws_lambda -c .lithops/$(ENV)/config.predict.yaml ttc-predict-dev

gate-a:
	cd $(REPO_ROOT) && uv run python scripts/predict_container_smoke.py

clean:
	rm -rf $(RENDERED)
