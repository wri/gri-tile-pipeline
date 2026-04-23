# Manual runbook — updating `tof-output`'s bucket policy in the `wri` account

Terraform does not apply changes to the `wri` account. Any modification to
`tof-output`'s bucket policy is done manually via this runbook, using the
statement JSON that Terraform emits as an output.

Run this after any of:

- First-time stand-up of the `land-research` env stack (grant the new Lambda role).
- A change to `var.output_bucket_prefixes` or `var.role_name` in land-research.
- A re-created Lambda role (new ARN).
- A rollback to a prior policy snapshot.

---

## Prerequisites

- `<wri_aws_profile>` profile (or whichever `wri` profile you use) with current
  credentials — verify with `aws sts get-caller-identity --profile <wri_aws_profile>`.
- `jq` installed (`brew install jq`).
- Most recent `terraform apply` on `infra/terraform/envs/land-research` has
  succeeded (it emits the statement JSON you'll use below).
- `uv` available (for running `scripts/merge_tof_output_policy.py`).

---

## Step 1 — Snapshot the current policy (rollback file)

```bash
SNAPSHOT=~/tof-output-policy-snapshot-$(date +%F-%H%M).json

aws s3api get-bucket-policy --bucket tof-output --profile <wri_aws_profile> \
    | jq -r .Policy | jq . > "$SNAPSHOT"

echo "Snapshot: $SNAPSHOT"
```

If `get-bucket-policy` returns `NoSuchBucketPolicy`, the bucket has no policy
yet — create an empty stub:

```bash
echo '{"Version":"2012-10-17","Statement":[]}' | jq . > "$SNAPSHOT"
```

**Do not commit `$SNAPSHOT` to git.** Keep it in your home directory or a
secure notes app.

---

## Step 2 — Generate the statements Terraform wants appended

```bash
terraform -chdir=infra/terraform/envs/land-research output -raw cross_account_policy_statements_json \
    | jq . > /tmp/tf-cross-account-statements.json

cat /tmp/tf-cross-account-statements.json
```

You should see a JSON array with two `Sid`s — `CrossAccountListBucket` and
`CrossAccountReadWrite` — each granting the Lambda role (`lithops-execution-role`)
the relevant S3 actions on `tof-output` (or the scoped prefixes, if you set
`output_bucket_prefixes`).

Sanity check the role ARN:

```bash
terraform -chdir=infra/terraform/envs/land-research output -raw lambda_role_arn
# should end in :role/lithops-execution-role
```

---

## Step 3 — Merge

```bash
uv run python scripts/merge_tof_output_policy.py \
    --current "$SNAPSHOT" \
    --append /tmp/tf-cross-account-statements.json \
    --out /tmp/tof-output-policy-merged.json
```

The script:

- Dedupes by `Sid` — re-running the runbook replaces old copies of the
  Lambda-role statements instead of stacking duplicates.
- Validates the merged policy is under S3's 20 KB bucket-policy size limit.
- Prints an `Added` / `Replaced` / `Untouched` summary.

Any pre-existing statement whose `Sid` does **not** start with `CrossAccount…`
is preserved verbatim in the `Untouched` list.

---

## Step 4 — Diff and review

```bash
diff <(jq -S . "$SNAPSHOT") <(jq -S . /tmp/tof-output-policy-merged.json) | less
```

**Expect only additions (`>` lines).** If you see removals (`<` lines) for
statements you didn't intend to touch, **stop**. Investigate:

- Is the removed SID a legacy statement that this runbook should preserve?
  Re-run Step 1 on a freshly fetched policy and re-check.
- Did someone modify the policy between the snapshot and now? Re-snapshot.

---

## Step 5 — Apply

```bash
aws s3api put-bucket-policy \
    --bucket tof-output \
    --policy file:///tmp/tof-output-policy-merged.json \
    --profile <wri_aws_profile>
```

No output on success. AWS rejects the call if the JSON is malformed or violates
policy-size limits.

---

## Step 6 — Verify

**Count statements:**

```bash
aws s3api get-bucket-policy --bucket tof-output --profile <wri_aws_profile> \
    | jq -r .Policy | jq '.Statement | length'
```

Expect: `(count from $SNAPSHOT) + (new-or-replaced statements from Step 3)`.

**Smoke test from the land-research side** — requires `<land_research_aws_profile>` profile and
`AWS_PROFILE=<land_research_aws_profile>`:

```bash
AWS_PROFILE=<land_research_aws_profile> LITHOPS_ENV=land-research \
    uv run python scripts/predict_lambda_smoke.py
```

The smoke script exercises both read (ARD) and write (predicted TIF) against
`tof-output`. `SMOKE TEST PASSED` confirms the grant works end-to-end.

---

## Step 7 — Rollback (if needed)

If the new policy breaks anything:

```bash
aws s3api put-bucket-policy \
    --bucket tof-output \
    --policy file://"$SNAPSHOT" \
    --profile <wri_aws_profile>
```

Re-verify with the smoke script or a manual `aws s3 ls` probe.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `NoSuchBucketPolicy` on Step 1 | Bucket has never had a policy | Use the empty-stub snapshot shown in Step 1. |
| `An error occurred (AccessDenied)` on `put-bucket-policy` | `<wri_aws_profile>` profile lacks `s3:PutBucketPolicy` on `tof-output` | Coordinate with the bucket owner to get the permission; do not proceed. |
| Merge script exits `merged policy is N bytes, exceeds …` | Cumulative statements too large | Audit the `Untouched` list — clean up stale statements in a separate manual edit before re-merging. |
| Smoke test fails with 403 after apply | Role ARN in the applied policy doesn't match the current Lambda role | Re-run Step 2 against a freshly applied `land-research` stack, confirm the role ARN output matches `aws lambda get-function-configuration --function-name ttc-predict-dev --query Role`. |
