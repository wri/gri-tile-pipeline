ENV ?= land-research
RENDERED := ../.lithops/$(ENV)
REPO_ROOT := ..

.PHONY: help render clean gate-a docker-gitsha \
        build-loaders-usw2 build-loaders-euc1 build-s1 build-predict build-all \
        deploy-predict update-lambda-predict

help:
	@echo "Targets (pass ENV=<env>, default: $(ENV)):"
	@echo "  render               Render Lithops configs from Terraform outputs"
	@echo "  gate-a               Build predict image locally + verify vs golden tile"
	@echo "  build-loaders-usw2   lithops runtime build ttc-loaders-dev (us-west-2)"
	@echo "  build-loaders-euc1   lithops runtime build ttc-loaders-dev (eu-central-1)"
	@echo "  build-s1             lithops runtime build ttc-s1-dev"
	@echo "  build-predict        lithops runtime build ttc-predict-dev"
	@echo "  build-all            All four runtimes"
	@echo "  deploy-predict       Build + push predict + pin Lambda(s) to the new digest"
	@echo "  update-lambda-predict   Re-pin predict Lambda(s) to ECR :latest (no rebuild)"
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

# Eager Lambda deploy. Lithops's `runtime deploy` builds + pushes but
# only calls CreateFunction on first invoke; if the Lambda already
# exists from a prior deploy, it stays pinned to its original image
# digest no matter how many times we push to ``:latest`` in ECR.
# update-lambda-predict closes that gap by explicitly calling
# UpdateFunctionCode against every Lambda whose image URI references
# the predict ECR repo. Effect: `make deploy-predict` is now actually
# self-contained — your next `sentinel-batch-flow` invocation will
# run the new code without manual `aws lambda update-function-code`.
deploy-predict: build-predict update-lambda-predict
	cd $(REPO_ROOT) && uv run lithops runtime deploy \
	    -b aws_lambda -c .lithops/$(ENV)/config.predict.yaml ttc-predict-dev

# Re-pin every predict Lambda to the latest digest in ECR. Standalone
# target so it can be invoked without rebuilding (e.g. after a manual
# `aws ecr put-image` to roll back to a known-good digest). Picks up
# the AWS_PROFILE from the calling shell — defaults to the
# AWSAdministratorAccess-058755926933 used elsewhere in this repo.
update-lambda-predict:
	@AWS_PROFILE=$${AWS_PROFILE:-AWSAdministratorAccess-058755926933} ; \
	REGION=us-east-1 ; \
	echo "==> Looking up latest digest in ECR (predict)" ; \
	REPO=$$(aws ecr describe-repositories --region $$REGION \
	    --query "repositories[?contains(repositoryName, 'ttc-predict-dev')].repositoryName" \
	    --output text | tr '\t' '\n' | head -1) ; \
	if [ -z "$$REPO" ]; then echo "ERROR: no ttc-predict-dev ECR repo" >&2; exit 1; fi ; \
	DIGEST=$$(aws ecr describe-images --region $$REGION --repository-name $$REPO \
	    --query 'sort_by(imageDetails, &imagePushedAt)[-1].imageDigest' --output text) ; \
	if [ -z "$$DIGEST" ] || [ "$$DIGEST" = "None" ]; then \
	    echo "ERROR: no images in $$REPO" >&2; exit 1 ; \
	fi ; \
	ACCT=$$(aws sts get-caller-identity --query Account --output text) ; \
	URI="$$ACCT.dkr.ecr.$$REGION.amazonaws.com/$$REPO@$$DIGEST" ; \
	echo "    repo:   $$REPO" ; \
	echo "    digest: $$DIGEST" ; \
	FNS=$$(aws lambda list-functions --region $$REGION \
	    --query "Functions[?contains(Code.ImageUri, '$$REPO')].FunctionName" \
	    --output text) ; \
	if [ -z "$$FNS" ] || [ "$$FNS" = "None" ]; then \
	    echo "    no Lambda functions yet — Lithops will create one on first invoke" ; \
	    exit 0 ; \
	fi ; \
	for FN in $$FNS; do \
	    echo "==> Pinning $$FN" ; \
	    aws lambda update-function-code --region $$REGION \
	        --function-name "$$FN" --image-uri "$$URI" \
	        --no-cli-pager --output text > /dev/null ; \
	    until [ "$$(aws lambda get-function --region $$REGION --function-name $$FN \
	        --query 'Configuration.LastUpdateStatus' --output text)" = "Successful" ]; do \
	        sleep 3 ; \
	    done ; \
	done ; \
	echo "==> Done"

gate-a:
	cd $(REPO_ROOT) && uv run python scripts/predict_container_smoke.py

clean:
	rm -rf $(RENDERED)
