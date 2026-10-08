# Purge Simulator: Using the App Well

A practical guide to the standalone app (`python app.py`): how to set up and run a job
with as few mistakes as possible, how to get good answers out of the assistant without
burning API credit, and what to check before a number goes to a client.

Everything here is based on how the app actually works today (main branch, October 2026).
Where a rule comes from a specific behavior in the code, the guide says so.
This guide is also the **Guide** tab in the app, so it's always one click away.

---

## 1. The one idea that saves the most money

**Only the assistant panel costs API credit. Everything else in the app is free.**

Importing files, the Job setup form, the Pre-run check, editing inputs, Run, the charts, the
Pipeline profile slider and every report export all run on your own computer. They never
call the Claude API. The USGS elevation lookup goes out to the internet, but it is a free
government service, not the Claude API.

So the cheapest, most reliable way to use the app is:

- **Do the routine work yourself with buttons**: import, answer the setup questions, run,
  look at the charts, export.
- **Use the assistant for judgment and for batches of work**: setting up a job from a
  description, explaining why a run behaves the way it does, comparing several options in
  one go, spotting problems you might miss.

---

## 2. Starting and stopping

1. Open PowerShell and go to the app folder:
   `cd C:\Users\kevin\Purge-Simulator-App`
2. Get the latest version: `git pull`
3. Start it: `python app.py`
   The app opens in your browser. The PowerShell window has to stay open while you work.
4. When you're done, click **Quit** in the app (or press Ctrl+C in PowerShell).

Things to know:

- **Quitting erases the assistant conversation.** The chat lives only in memory while the
  app runs. Anything worth keeping should go into the scenario's notes and be saved (see
  section 7). Refreshing the browser tab is fine; only Quit or closing PowerShell loses it.
- **Unsaved scenarios are lost on Quit too.** A freshly imported scenario is not saved until
  you click **Save as…**.
- If `git pull` complains about local changes, don't force anything. Copy the message and
  ask in the project.

---

## 3. How the API is actually used (so you can control the cost)

This is what happens in the code (`purge_sim/app/assistant.py`) every time you press Send:

- **Each message starts a loop.** Claude reads your message, then calls the app's tools
  (read the scenario, change inputs, run, sweep, and so on). **Every tool step is another
  API call**, up to 25 per message. A simple question might be 1 or 2 calls; "set up this
  job and run it" is typically 4 to 8.
- **Every call resends the whole conversation so far.** The system instructions, the tool
  list, every earlier message and every earlier tool result go back to Claude each time.
  So a long conversation gets more expensive with every message, even if the new question
  is short.
- **Prompt caching softens this, but only for a few minutes.** The app marks the
  conversation for caching, so a repeated beginning of the conversation is billed at a much
  lower rate. The cache expires after about five minutes of no activity, so a conversation
  you come back to after lunch is billed at full price on the next message.
- **The assistant is set to think hard on every reply** (high effort, extended thinking).
  That is good for engineering judgment and wasteful for "what's the drive cap set to?"
  (which you can read on the Inputs tab for free).
- **Big tool results are big bills.** A run summary is small. A full time series (up to
  600 rows), a full-route pressure profile, or a long elevation sample is large, and it
  stays in the conversation and is resent with every later call.
- **Simulation time is free.** While a run or sweep is going, no tokens are being used;
  the assistant is just waiting on your computer.

### Habits that cut API usage

| Do this | Instead of this | Why |
|---|---|---|
| Click **New chat** when you switch to a different job or a different question | One endless conversation all day | Old messages are resent on every call |
| Put the whole job in one message: size, wall, grade, product, MOP basis, drive cap, launch/stop MP, where the liquid leaves, speeds | Drip-feeding one fact per message | One `set_job_setup` call instead of many rounds |
| Ask for a sweep: "compare drive caps of 450, 500, 550 and 600" | "Try 450." … "Now try 500." … | A sweep (up to 8 values) is one tool call and doesn't disturb your open scenario |
| Read the Charts, Overview and Pipeline profile tabs yourself | "Show me the pressure at every step" | Long tables go into the conversation and are resent |
| Ask focused questions: "Why does the speed drop at MP 18?" | "Tell me everything about this run" | Less reading, shorter answers |
| Use Run, Pre-run check and the form buttons yourself for routine changes | Asking the assistant to change one number and rerun | Buttons are free |
| Keep working in one sitting when you're mid-conversation | Leaving a conversation idle and coming back much later | The cache expires after about five minutes |
| If you hit "Stopped after 25 tool rounds", split the request into two | Repeating the same big request | It hit the per-message limit; repeating it pays again |

### Choosing the model

The model is set in **Settings** (default `claude-opus-5-5`). That default is the right choice
for job setup and for anything you'll rely on in a bid. If you are doing a long session of
routine "run this, compare that" work, you can switch to `claude-sonnet-5-5` in Settings,
which costs less per message, and switch back for setup and review. Don't put any other
model name in there unless you've been told it works; a wrong name shows "Model … wasn't
found".

### Errors you might see

- **"Add your Claude API key in Settings"**: no key yet. Paste it in Settings.
- **"The Claude API key was rejected"**: the key is wrong or was revoked.
- **"Couldn't reach the Claude API"**: internet, or an out-of-date install. Run
  `pip install -r requirements.txt` once in the app folder; that fixes the known cause.
- **"Rate limited"**: wait a minute and send again.

---

## 4. Setting up a new job: the reliable order

Follow these steps in order. Most mistakes come from doing them out of order.

### Step 1: Import the data

Click **Import data…** and leave the data type on **Detect automatically**. The app tells
these apart on its own:

- **Rosen ILI Excel**: elevation, per-joint MOP, pipe geometry, pump stations, check valves
  and BPCV are read from the file.
- **Point-by-point (PxP) pressure sheet** (operator format, e.g. P66 GL-09): same kind of
  data as an ILI.
- **Elevation profile**: KMZ/KML, GPSVisualizer TXT, plain milepost/elevation text, or a
  client Excel profile.

Alternatively, attach the file in the assistant panel with the 📎 button and describe the
job in the same message (see section 5). The import itself is still free; only the message
costs.

**Check right after importing** (open the Notes on the Overview tab):

- **Which format it detected.** If an ILI was read as a profile or vice versa, re-import with
  the data type picked by hand.
- **Where the elevation came from.** A KMZ with no elevations is looked up in USGS 3DEP at
  250 ft spacing (US only, needs internet). Look at the elevation line on the Overview tab
  for the number of points, how many were interpolated, and any warning flags (for example
  dead-flat stretches that may be a river or HDD crossing).
- **Route direction.** KMZ routes are often drawn backwards. If MP 0 is at the wrong end, fix
  it with the first Job setup question ("Pig runs from the file's far end back toward its
  start"), not by editing mileposts.
- **For a PxP sheet:** mileposts are the sheet's distance from the line's origin, so the
  purge may start well past MP 0. MOP is the sheet's MOP column, which can be lower than its
  MOP Limit column. The product is not set from the sheet's specific gravity; you still pick
  it.

### Step 2: Answer the Job setup questions

Open the **Job setup** tab. Answer everything you know, then click **Apply**.

- **Required:** pipe size, product, MOP. The app will not guess these.
- **Blank questions use a default.** Defaults marked **assumed** are typical values (for
  example standard wall thickness, the product library's SG and viscosity, 50 psig at
  tankage, a 3 mph target speed), not facts about this job. Every assumption is listed after Apply and written into the scenario notes.
  **Read that list and either confirm or answer each one.** An unconfirmed assumption is the
  most common source of a wrong result.
- **Where the liquid really leaves the line.** If the pig stops at a block valve but the
  product keeps going to a tank farm or pump further on, answer "Where the liquid actually
  leaves the line". The app then holds the right back pressure at the pig stop. Leaving it
  blank when the liquid does go further makes the purge look easier than it is (on Laurel the
  difference is about 400 psig vs 50 psig at the pig stop).
- **Volatile products** (butane, NGL): the delivery pressure must stay above vapor pressure +
  100 psi. The Pre-run check flags it if it doesn't.

### Step 3: Run the Pre-run check

Click **Pre-run check** (free, under a second). It tells you, before any simulation:

- the job type: speed-capped, drive-capped, friction-dominated, laminar, gravity-assisted,
  or pump/BPCV controlled;
- the drive pressure needed at launch and at the worst point, compared with your drive cap,
  and what speed the cap allows there;
- the highest pack pressure that keeps every joint under MOP;
- the minimum N2 for pack-and-coast at target speed;
- downhill stretches steep enough to run away from the pig.

If the check says the target speed can't be held under the drive cap, fix that now (lower
the speed, or confirm a higher cap) instead of finding out after a long run.

### Step 4: Things the questions don't cover (Inputs tab)

Use the **Inputs** tab for:

- **Lines that change size.** The Job setup sets one uniform pipe size; then edit the pipe
  segments here.
- **Which detected stations really pump.** An ILI import marks pump stations and gives each
  a 30 psig suction default. Whether each one is a real pump, a BPCV, or nothing during the
  purge is still your call. This is the single biggest judgment the software can't make for
  you.
- **The BPCV and booster stations.**

> **Watch out:** clicking **Apply** on the Job setup tab again rebuilds the inputs from the
> answers. If you have split the pipe into several sizes on the Inputs tab, Apply puts it
> back to one uniform size (it warns you when it does). So finish the Job setup first, then
> make Inputs-tab edits, and re-check the pipe segments if you ever Apply the setup again.

### Step 5: Run

Click **Run**. Long routes can take a few minutes. Then read the **Overview** tab first.

---

## 5. Getting the most out of the assistant

### What it can and can't do

It can: list and open scenarios, read every input and the elevation/MOP data, answer the
Job setup questions, run the Pre-run check, change inputs, run the simulation, sweep one
input across up to 8 values, read results and the pressure profile at any moment, look up
pipe sizes and Barlow pressures, re-fetch elevation from USGS, and save a new scenario.

It can't (by design): overwrite a bundled or existing scenario, type in elevations or MOP
values, set a booster suction floor below 100 psi, or save unless you ask. Every edit it
makes goes through the same checks as the form, and it changes the same scenario you see
on screen.

### Write messages like a job ticket

Good first message for a new job (with the KMZ attached):

> 6 inch diesel line, pig launches at MP 0 and stops at the block valve at MP 22, but the
> diesel goes on to tankage at MP 31. Wall 0.280, X52. MOP 900 at all points. Drive capped
> at 500. Target 3 mph, lean strategy.

The assistant maps that onto the Job setup questions in one step, reports the assumptions
it had to make, runs the Pre-run check and then the simulation. Things it is told never to
do: invent pipe size, product or MOP, or invent elevations. If one is missing, it will ask.

### Good questions to ask

- "What kind of job is this, and what limits it?"
- "Why does the pig slow down between MP 18 and 22?"
- "Compare N2 budgets of 2.0, 2.5 and 3.0 MMSCF with pack-and-coast."
- "Where is the tightest MOP margin and how close does the run get?"
- "Which of the detected stations should really pump? Explain your reasoning."
- "Is there anything in this setup you'd question before I send it?"

### Things to watch for in its answers

- **Every number should come from a run in this conversation.** It is told to say which run
  or scenario a number came from, and to say when it is estimating. If it doesn't, ask.
- **Its classification of stations and job type is a recommendation, not a decision.** You
  make the final call.
- **It changes your open scenario.** Its edits show up on screen right away, unsaved. The
  sweep tool is the exception: sweeps leave the open scenario alone.
- **A new chat starts with no memory of earlier chats.** If a decision matters, it must be
  in the scenario notes (which the assistant reads) and saved.

---

## 6. Reading results: what must be zero

Open the **Overview** tab after every run. A clean run shows a green line: *No venting, no
slack-line risk, no MOP violations.* Anything else is listed in red. The engineering rules
behind these:

1. **N2 vented must be 0.** Venting is a last-resort safety net in the engine, never a plan.
   Any vented SCF means the drive or booster settings need fixing.
2. **Slack-line risk steps should be 0.** If not, the drive can't keep the liquid above its
   minimum pressure at those steps. Accepting slack line is a deliberate engineering decision
   (lowering the minimum liquid pressure with sign-off), never something to ignore.
3. **MOP violation steps must be 0.** Check the worst MOP margin and where it is.
4. **The run must complete.** "Did not complete" with a reason means the result is not a
   valid plan.

Also check:

- **"Inputs changed since this run"** (yellow): the results on screen are from before your
  last edit. Run again before reading them.
- **Pig speed:** average and minimum against your target and minimum speed.
- **Booster plan:** which sites were used and why.
- **Charts tab, elevation:** tick *Show elevation profile on each chart* to add the ground
  elevation (ft) on a right-hand axis of every chart. On the milepost charts it is the profile
  along the route; on the time charts it is the ground under the pig at that moment. The
  pressure chart also gets *HGL at pig face*: the liquid head right at the pig (elevation +
  face pressure ÷ (0.433 × SG)), on the same feet axis. The box stays ticked between runs.
- **Charts tab, Hydraulic grade line:** the whole route at one moment of the run, all in feet:
  ground elevation, the liquid hydraulic grade line ahead of the pig, the N2 pressure behind
  the pig shown as equivalent liquid head, and the MOP grade line (elevation + MOP as head).
  Drag the slider to move through the run. Where the HGL comes close to the ground the liquid
  pressure is low (slack-line risk); where it reaches the MOP line the pipe is at MOP. The
  head uses the run's fluid SG.
- **Pipeline profile tab:** scrub through the run and watch the N2 pressure, liquid pressure
  and MOP lines, especially over peaks and near pump stations.
- **Map tab:** the route on a basemap with the pig's progress. A scenario needs route
  coordinates for this; the bundled client jobs don't have them yet, so add the job's KMZ
  once from the Map tab and save.

---

## 7. Saving, variants and reports

- **Save as… with a name that says what changed**, e.g. `GL09 12in diesel 500psi cap 3mph`.
  Saving never overwrites: a name that already exists is refused. The bundled client-job
  scenarios are never written to.
- **Write the "why" into the notes** when you save: why that drive pressure, why that MOP
  basis, who confirmed the assumptions. The Job setup block in the notes is kept up to date
  automatically; your own reasoning is not.
- Saved scenarios go to `PurgeSimScenarios` in your home folder (shared with the older
  desktop app). Reports go to a new dated folder under `PurgeSimOutputs` every time, so
  earlier variants are never overwritten.
- **Reports tab, Download all reports (zip):** one click writes every deliverable for the
  run and downloads them as one zip: the client web page, both xlsx reports, the run log,
  the profile animation GIF and the scenario file that produced the run. Untick *Include
  the profile animation* when you want it fast (the GIF is the slowest part).
- **Client web page (html):** a single file the client opens in any browser, with no
  internet and nothing to install. It shows the headline results and any flags, a pipeline
  profile they can scrub or play through the run (N2 and liquid pressure, MOP, elevation,
  pump station status), the four charts with hover readouts, the job basis, stations and
  valves, and the 50-point FILL REPORT table. It prints cleanly (Ctrl+P, or save as PDF).
  Your scenario notes are **left off** unless you tick *Put the scenario notes on the client
  web page*, because notes often carry internal tender reasoning.
- **One at a time:** *Client web page*, *Client report* (FILL REPORT table + charts, xlsx)
  for the client, *Full technical report* (7 sheets) for your own record, *Run log*, and the
  *Profile animation* GIF.
- Exports describe the latest run, using the inputs it was run with, even if you edited
  the scenario since. Check the yellow "Inputs changed" warning is gone before exporting
  if you want your edits in the reports.

---

## 8. Before a number goes to a client

- [ ] Pipe size, wall, grade, product and MOP basis are confirmed, not assumed.
- [ ] Every item in the Job setup assumptions list was confirmed or answered.
- [ ] Route direction is right (MP 0 is the launch end).
- [ ] The liquid exit is right: pig stop vs where the product actually leaves.
- [ ] Elevation source checked; interpolated gaps and flat stretches looked at.
- [ ] Station roles (pump / BPCV / nothing) are your decision, not the import default.
- [ ] Pre-run check job type makes sense for the line.
- [ ] Run completed with 0 vented, 0 slack-line steps, 0 MOP violations.
- [ ] Results are from the current inputs (no "Inputs changed" warning).
- [ ] Scenario saved under a new name with the reasoning in its notes.
- [ ] Reports exported from that saved run.

---

## 9. Known limitations to keep in mind

- **Bundled scenarios may not reproduce earlier deliverables exactly.** Several of the
  regression baselines don't match what the current code produces, most likely because
  earlier runs used engine changes that were never committed. Don't assume rerunning a
  bundled job gives the numbers you sent before; compare against the delivered report.
- **Drive vs. pump-held pressure in the profile chart.** The pig solver uses simpler exit
  hydraulics than the displayed station pressures, so the N2/liquid interface can sit below
  the pump-held line on the chart. This is a known open engine item.
- **The Laurel and GL-08 scenarios** carry a liquid exit past the pig stop that the engine
  doesn't read yet. If you reuse them, set the liquid exit through Job setup.
- **Elevation lookup is US only.**
- **Station classification and the final strategy** (lean vs pack-and-coast, booster
  plan) are still engineering judgment. The app and assistant help you make the call; they
  don't make it.
