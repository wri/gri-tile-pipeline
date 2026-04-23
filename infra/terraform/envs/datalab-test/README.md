# datalab-test — throwaway env for exercising the stack against `dl-user`

Purpose: validate the Terraform module graph, IAM role wiring, state-bucket
creation, and render pipeline against the datalab account while waiting on
elevated permissions in `land-research`. **Not** a long-lived env.

## Differences from `envs/land-research/`

| Concern | datalab-test | land-research |
|---|---|---|
| Backend | Local (no S3 state, no bootstrap needed) | Remote S3 + DynamoDB |
| Role name | `lithops-execution-role-dltest` | `lithops-execution-role` |
| State bucket prefix | `gri-ttc-dltest-state-<region>` | `lithops-ttc-<region>` (default) |
| Default tags | Include `Ephemeral = true` | Production tagging |
| wri cross-account grant | Skipped entirely — outputs emit but are not applied | Applied manually via `docs/manual_wri_policy_update.md` |

## Run

```bash
cd infra/terraform/envs/datalab-test
cp terraform.tfvars.example terraform.tfvars       # optional; defaults are fine

AWS_PROFILE=dl-user terraform init                  # local backend, no -backend-config flags
AWS_PROFILE=dl-user terraform apply
AWS_PROFILE=dl-user terraform output                # sanity-check outputs
```

Render Lithops configs from the outputs (writes to `.lithops/datalab-test/`):

```bash
AWS_PROFILE=dl-user make -C infra render ENV=datalab-test
grep -l '\${' .lithops/datalab-test/*.yaml && echo "BAD" || echo "OK"
```

Gate A (local container parity) is env-independent and can run here too:

```bash
make -C infra gate-a
```

## Teardown

```bash
AWS_PROFILE=dl-user terraform destroy
rm -rf .terraform .terraform.lock.hcl terraform.tfstate* terraform.tfvars
```

Once `land-research` perms are in place, delete the entire
`envs/datalab-test/` directory and the rendered `.lithops/datalab-test/`
configs — there's nothing left to carry forward.

## What this env does NOT exercise

- Pushing ECR images in datalab (Lithops `runtime build`). Orthogonal to
  the Terraform stack; can be run separately if dl-user has ECR perms.
- The cross-account grant against `tof-output` in the wri account —
  datalab is not the account we're granting to.
- Lambda parity (`PARITY_LAMBDA=1 pytest …`) — requires a deployed Lambda
  that can read `tof-output`, which isn't happening here.
