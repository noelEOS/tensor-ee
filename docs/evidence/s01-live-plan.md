# S01 live plan: public copies of the trained weight assets in `tensor-ee`

Status: **approved and run; passed.** The user approved this plan as written on 2026-10-15, and
the lead ran it that day (01:53–02:13 UTC) at revision `2602af3`: 1,164 calls of the 1,200 cap
(the 1,163 below, plus one destination-root listing made twice), with zero exports or tasks.
104 copies equal to their sources and readable by all users, and 220 layers' sampled rows
exact (largest absolute error 0). The second account's checks reported 2 of 2 and 104 of 104.
The results, the two stops and how they were resolved are in [s01-live-run.json](s01-live-run.json).
The run's credentials were the `tensor-ee` owner's Application Default Credentials, supplied to
the run's process alone through `GOOGLE_APPLICATION_CREDENTIALS`. The plan below is otherwise
unchanged from what the user approved.

Part 1 of S01 (development archive: `docs/tasks/S01-public-weight-assets.md`) is offline: the tool
([`tools/public_weights.py`](../../tools/public_weights.py)), the public manifests (its
transform, pinned by SHA-256 in [the link record](s01-weights-link.json)), the link proof and the
access check ([`s01-public-access-check.js`](s01-public-access-check.js)). This plan is the one
live step between Part 1 and Part 2. It has the form of [L06c's plan](l06c-live-run-plan.md).

The run copies the 104 weight tables, server side and unchanged, from the original project into
`tensor-ee`, makes each copy readable by all users, and checks the copies. It starts no export
and no batch task. The original project's ID appears nowhere in this plan: the tool reads it from
the tracked manifests' asset paths, and the run's files, which name it, stay outside the
repository.

| Weights | Manifest | Shard assets | Destination | Upload payload |
|---|---|---|---|---|
| SegWater U-Net | `data/manifests/segwater_r18_weight_manifest.json` | 29 | `projects/tensor-ee/assets/` (the root) | 199,631,866 bytes |
| Swin-T/UPerNet | `data/manifests/m12_trained_weights_manifest.json` | 75 | `projects/tensor-ee/assets/upernet/trained/swint_step6000_last_fp32_1f398d3256ba/` | 516,301,722 bytes |

Each destination is its source's path with the prefix `projects/<original>/assets/` replaced by
`projects/tensor-ee/assets/`; the rest of the path, timestamped names included, is unchanged.

## Accounts and prerequisites

Two Google accounts take part. **The `tensor-ee` owner** owns `tensor-ee`, the project created for
public assets. **The original project's owner** owns the original project, and is the second
account of the access checks below: its registered project of its own is the original project,
and it holds no role on `tensor-ee`.

| Prerequisite | State | Checked by |
|---|---|---|
| `tensor-ee` is registered for Earth Engine, with the Earth Engine API enabled | set up by the user | the preflight's listing of `projects/tensor-ee/assets` |
| The run's Application Default Credentials are the `tensor-ee` owner's (this laptop's gcloud) | **met: set up by the user on 2026-10-15** | the preflight's listing, and the pilot's `createFolder` and `copyAsset` |
| The `tensor-ee` owner can read the 104 source assets: read access on the original project, granted by the user (a viewer role on the original project, or read access on those assets, would also do) | **met: granted by the user on 2026-10-15** | preflight step 1: `getAsset` on every source, then its feature count |
| The tool initializes Earth Engine with project `tensor-ee` | in the tool (`ee.Initialize(project="tensor-ee")`) | the preflight's listing |
| The second account (the original project's owner) holds no role on `tensor-ee` | **met: confirmed by the user on 2026-10-15 on `tensor-ee`'s IAM page** | before the pilot check: recheck `tensor-ee`'s IAM page if any IAM change has happened since |
| The second account can open the Code Editor: Earth Engine's terms accepted and a registered project of its own | the original project | the user, when running the access check |

A role on the original project does not affect the access test, which is about `tensor-ee` only.
The `tensor-ee` owner's credentials run every call: `copyAsset` needs one identity that can both
read the sources and create assets in `tensor-ee`, and granting the second account a role on
`tensor-ee` would end its use as the access test.

The run identity is never inspected from credential files. Because the second account holds no
role on `tensor-ee` (confirmed on its IAM page), a preflight that can list `tensor-ee`'s asset
root, and the pilot's successful `createFolder` or `copyAsset`, show that the run is not using the
second account.

When a prerequisite is missing, the call that needs it fails and the run stops at it, its reason
naming the prerequisites: a source the run identity cannot read stops preflight step 1 with the
two source prerequisites; a destination root that cannot be listed, or a destination whose
existence cannot be checked, stops with the `tensor-ee` prerequisites (registration, the API, the
owner's credentials, project `tensor-ee`).

## Commands

From a checkout at the accepted Part 1 revision, with the main checkout's `.venv` and
`PYTHONPATH=.`. `<state>` is a new empty directory outside the repository, or under the
git-ignored `release_runs/`; the tool refuses any other path inside the checkout, because the
run's files name the source assets. `<resources>` is the main checkout, which holds the trained
checkpoints the row check reads.

```sh
python tools/public_weights.py --plan --markdown          # offline: the tables below
python tools/public_weights.py --run --state-dir <state> --resources <resources> --max-calls 1200
#   stops after the pilot (AWAITING_PILOT_CHECK, exit status 2)
#   the user's pilot access check reports 2 of 2 passed, then:
python tools/public_weights.py --run --state-dir <state> --resources <resources> --pilot-passed
python tools/public_weights.py --status --state-dir <state>   # at any time; no cloud call
```

`--run` checks both checkpoints' SHA-256 (G1's pins) before it initializes Earth Engine. The
first invocation records the caps in the state file (`--max-calls 1200`, and the default 3,600 s
per invocation); every resume keeps them, so the resume commands repeat neither flag. A resume
given `--max-calls` or `--max-seconds` replaces that cap, and the state's history records the
change.

**The call budget: the user's decision of 2026-10-15.** Keep the feature counts (the run's 1,163
calls below) and cap the run at 1,200 calls (`--max-calls 1200`), a margin of 37 for resumes. This
sets the budget only; it is not the user's approval of this plan.

## Every call, with its count

The run makes these calls in this order, one at a time. `computeValue` is the call `getInfo`
makes: `ee.FeatureCollection(id).size()` for a feature count, and one dictionary of sampled rows
per layer for the row check (`native_ee.weight_assets.verify_asset_rows`, 3 rows per layer).

| Phase | Call | Count | What |
|---|---|---|---|
| Preflight | `getAsset` | 104 | each source asset (step 1: readable by the run identity); type and sizeBytes |
| Preflight | `computeValue` | 104 | each source's feature count, size(), equal to its manifest rows |
| Preflight | `listAssets` | 1 | the destination root projects/tensor-ee/assets |
| Preflight | `getAsset` | 107 | each destination folder and asset: must not exist (HTTP 404) |
| Pilot | `createFolder` | 3 | the folders of the pilot's Swin-T shard |
| Pilot | `copyAsset` | 2 | the first shard of each manifest, allowOverwrite false |
| Pilot | `getAsset` | 2 | each copy: type and sizeBytes equal to its source's |
| Pilot | `computeValue` | 2 | each copy's feature count equal to its source's |
| Pilot | `getAssetAcl` | 2 | each copy's ACL before the change |
| Pilot | `setAssetAcl` | 2 | that ACL, readable by all users |
| Pilot | `getAssetAcl` | 2 | read back: readable by all users, owners, writers, readers unchanged |
| The rest | `copyAsset` | 102 | the remaining shards, allowOverwrite false |
| The rest | `getAsset` | 102 | each copy: type and sizeBytes equal to its source's |
| The rest | `computeValue` | 102 | each copy's feature count equal to its source's |
| The rest | `getAssetAcl` | 102 | each remaining copy's ACL before the change |
| The rest | `setAssetAcl` | 102 | that ACL, readable by all users |
| The rest | `getAssetAcl` | 102 | read back, as in the pilot |
| Row check | `computeValue` | 31 | row check, U-Net: 3 rows per layer, one layer per call, every error exactly 0 |
| Row check | `computeValue` | 189 | row check, Swin-T: 3 rows per layer, one layer per call, every error exactly 0 |
| **Total** | | **1163** | the Earth Engine calls the run counts and caps |

| Call | Count |
|---|---|
| `computeValue` | 428 |
| `copyAsset` | 104 |
| `createFolder` | 3 |
| `getAsset` | 315 |
| `getAssetAcl` | 208 |
| `listAssets` | 1 |
| `setAssetAcl` | 104 |
| **Total** | **1163** |

**The exact count is 1,163 calls**, plus one `algorithms.list` call that `ee.Initialize` makes in
each invocation (two for the run as planned: the pilot and the rest; one more per resume). There
are **zero exports and zero batch tasks**. `ee.data.setMaxRetries(0)` is local and contacts
nothing. Every call reads, except 3 `createFolder`, 104 `copyAsset` and 104 `setAssetAcl`, all in
`tensor-ee`.

Two differences from the brief's estimate of about 850 calls. The user kept both on 2026-10-15
(the call budget above):

- **Feature counts, +208.** `getAsset` does not report a table's feature count: the table assets
  recorded so far ([`segformer_b0-fullmodel-live.json`](segformer_b0-fullmodel-live.json)) answer
  type, name, ID, update time and `sizeBytes` only. So the tool counts features with `size()`, once
  per source (against its manifest rows) and once per copy (against its source). Without them
  (`--no-feature-counts`, in both `--plan` and `--run`) the run would make 955 calls and compare
  the copies on type and `sizeBytes` only. The `size()` counts are the feature check: a
  `featureCount` field is compared as well only if `getAsset` reports it for both the source and
  the copy.
- **The ACL before the change, +104.** `setAssetAcl` replaces an asset's whole IAM policy. The tool
  reads each copy's ACL first and sets the same ACL with `all_users_can_read`, so no owner, writer
  or reader entry is dropped, and the read-back must equal that ACL exactly. What the read-back
  shows has a boundary: `getAssetAcl` and `setAssetAcl` see the client's legacy-ACL projection of
  the asset's IAM policy, which is the owners, writers and readers and `all_users_can_read`. An
  asset-level binding on any other role would not be visible to the read-back, and
  `setAssetAcl`, which replaces the whole policy, would remove it. On an asset created seconds
  earlier none is expected, and no call is added to look. Project-level access is not touched
  and not affected, the `tensor-ee` owner's included. Whether an asset's owner entries round-trip
  through `setAssetAcl` cannot be shown offline: the pilot's read-back is the gate.

## Order, the pilot stop and the limits

1. **Preflight.** Step 1 reads every source (`getAsset`): each must be a table with a
   `sizeBytes`; then its feature count must equal its manifest's rows. Then the destination root
   must be listable, and each of the 3 destination folders and 104 destination assets must answer
   HTTP 404 (read from the error's HTTP status, never its text). An existing destination fails the
   run: nothing is overwritten.
2. **Pilot.** The 3 folders of the Swin-T pilot shard, then the first shard of each manifest
   (`segwater_r18_c34c75563208_1790603567_part_0000` and
   `upernet/trained/swint_step6000_last_fp32_1f398d3256ba/w_part_0000`): copied, each compared with
   its source (type, `sizeBytes`, the `size()` feature count, and `featureCount` when both sides
   report it), then its ACL read, set and read back. The run
   then **stops** (`AWAITING_PILOT_CHECK`) and does nothing more until the lead resumes it with
   `--pilot-passed` after the user's pilot check passes. `--pilot-passed` is refused at any other
   time.
3. **The rest.** The 102 remaining shards, each copied and compared as in the pilot (no folder is
   left to create), then the ACL of each, read back.
4. **The row check.** Every layer of both public manifests (31 + 189), each read from the copies
   through the public manifest and compared with the checkpoint's coefficients on 3 sampled rows.
   Every error must be exactly 0.

| Limit | Value |
|---|---|
| Calls | `--max-calls 1200` (the user's budget), recorded at the first invocation and kept by every resume, for the whole run across resumes (without the flag a new run records the plan's 1,163); each call is counted before it is made, and the run stops before a call past the cap |
| Time | `--max-seconds`, default 3,600 s per invocation, recorded and kept the same way; the run stops before a call past it |
| Retries | none: the client's retries are switched off before `ee.Initialize`, so initialization's reads are not retried either, and a failed call stops the run |
| Expected time | about 1,163 sequential calls, each usually under a few seconds: well under an hour (an estimate, not a guarantee) |

**Stops and failures.** The state file (`<state>/state.json`) is written before every call, with
the call, its target and its count, and after it with the result. A check that fails (an
existing destination, a copy that differs from its source, a source whose feature count is not
its manifest's rows, an ACL that does not read back, a row error that is not 0) **fails** the run:
`FAILED`, final, the lead reports to the user (see "After a `FAILED` run" below). A call that
raises, a cap or the time limit
**stops** it: `STOPPED`, with the call, its HTTP status and the reason recorded; the lead reads the
reason and decides whether to resume.

**Resuming is the lead's decision, and reconciles before it acts.** The tool never retries a call
by itself. A resume continues from the state file: every completed call's result is kept and
not repeated. A write whose outcome was not recorded (the call raised, or the process stopped
during it) is read before anything else: a folder or a copy is looked up (`getAsset`) and created
or copied only if it is absent; an ACL is read and set only if it still equals the ACL recorded
before the change. **Every call is counted when it is attempted**, so a run that stops and resumes
makes more calls than the table: each call that raised is made again on the resume, and each
reconciling read is one more. The 1,200-call cap, recorded at the first invocation and kept by
every resume, leaves 37 calls for that. If a run ever needs more, the lead resumes it once with a
higher `--max-calls`, which replaces the recorded cap from then on. `--status` shows the running
count and the cap. A refused ACL (`setAssetAcl` raising) stops the run, and the tool never grants
a role on the project instead. One process holds a run directory at a time (`run.lock`).

**After a `FAILED` run.** Nothing is retried and nothing is deleted. The lead stops and reports to
the user, with the reason from `--status`. The copies already made stay in `tensor-ee`, and a new
run refuses them, as it refuses every existing destination in its preflight. Deleting partial
copies is the user's decision, and the user's own action in `tensor-ee`; the tool never deletes.
An ACL that does not read back as expected is `FAILED`, not `STOPPED`: it is a state to inspect,
not a call to retry, so no resume continues it.

## The second account's checks

The access check is [`s01-public-access-check.js`](s01-public-access-check.js), generated from the
public manifests. For each asset it compares `ee.FeatureCollection(id).size()` with the manifest's
rows and prints one `PASS` or `FAIL` line; each asset is read on its own, one after another, with
an error callback, so a missing or unreadable asset prints a `FAIL` line and the check goes on.
The last line counts the passes. The user runs it in the Code Editor **signed in to the second
account** (the original project's owner, no role on `tensor-ee`): owner credentials can read
assets in any of their projects, so they prove nothing about public access.

1. **Pilot check**, at the pilot stop: recheck `tensor-ee`'s IAM page if any IAM change has
   happened since 2026-10-15, then run the script as committed (`PILOT_ONLY = true`). Expected:
   `S01 access check: 2 of 2 passed, 0 failed`. The Swin-T pilot shard lies inside three new
   folders whose ACLs the tool leaves unchanged, so this also shows that reading an asset by its ID
   needs only the asset's ACL. If the U-Net shard passes and the Swin-T shard fails, the lead stops
   and reports: a folder ACL would be a change to this plan.
2. **Full check**, after the run completes: set `PILOT_ONLY = false` and run it again. Expected:
   `S01 access check: 104 of 104 passed, 0 failed`.

## What never happens

- No export, no batch task, no `ee.batch` call.
- No deletion and no overwrite: `allowOverwrite` stays false, and the original assets are kept.
  Deleting them is a separate decision for the user.
- No IAM change on any project: the only permission changes are the 104 copies' own ACLs. No
  folder's ACL changes, nothing changes in the original project, and no role is granted to
  `allUsers` or `allAuthenticatedUsers` on a project.
- No change to the repository during the run: the public manifests replace the tracked ones in
  Part 2, after this run passes.

## Pass criteria and what is left behind

The run passes when `--status` reports `COMPLETED` with a summary of 104 copies equal to their
sources (type, `sizeBytes` and feature count), 104 copies readable by all users and read back, and
220 layers checked with the largest absolute error 0; and the user's full check reports 104 of 104.
Part 2 then starts.

Left behind: 104 table assets and 3 folders in `tensor-ee`, and the run directory. The run
directory names the original project's assets: it stays outside the repository and is never
committed. The lead's run record for Part 2 carries counts, booleans and SHA-256s only.
