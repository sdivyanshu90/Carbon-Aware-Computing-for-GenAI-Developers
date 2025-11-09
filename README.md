# Carbon Aware Computing for GenAI Developers

This repository contains learning materials and notes for the course **[Carbon Aware Computing for GenAI Developers](https://www.deeplearning.ai/short-courses/carbon-aware-computing-for-genai-developers/)**, an advanced program provided by **[DeepLearning.AI](https://www.deeplearning.ai/)** in collaboration with **[Google Cloud](https://cloud.google.com/)**.

## Introduction

This course teaches you how to design, train, and deploy Generative AI models with a focus on environmental sustainability. You'll learn to perform model training and inference jobs using cleaner, low-carbon energy in the cloud.

### What You'll Learn

  * **Query real-time electricity grid data:** Explore the world map and, based on latitude and longitude coordinates, get the power breakdown of a region (e.g., wind, hydro, coal) and its carbon intensity (CO2 equivalent emissions per kWh).
  * **Train a model with low-carbon energy:** Select a region with a low *average* carbon intensity to deploy your training job.
  * **Optimize with real-time data:** Go a step further by selecting the lowest carbon intensity region using *real-time* grid data from ElectricityMaps.
  * **Measure your impact:** Retrieve measurements of the carbon footprint for ongoing cloud jobs.
  * **Report your footprint:** Use the Google Cloud Carbon Footprint tool, which provides a comprehensive measure of your carbon footprint by estimating greenhouse gas emissions from your Google Cloud usage.

Throughout the course, you’ll work with the **ElectricityMaps API** for querying global grid information and use **Google Cloud** to run a model training job in a data center powered by low-carbon energy.

Get started, and learn how to make more carbon-aware decisions as a developer\!

-----

## Course Topics

This course is broken down into five key modules:

1.  **The Carbon Footprint of Machine Learning**
2.  **Exploring Carbon Intensity on the Grid**
3.  **Training Models in Low Carbon Regions**
4.  **Using Real-Time Energy Data for Low-Carbon Training**
5.  **Understanding your Google Cloud Footprint**

-----

<details>
<summary><strong>The Carbon Footprint of Machine Learning</strong></summary>

This foundational module unpacks the "why" behind the course, exploring the deep and often-overlooked connection between computation and carbon emissions. Before we can reduce our footprint, we must first understand how it's created. This topic provides the essential vocabulary and mental models needed to see ML not just as code, but as a physical process with real-world environmental impacts.

#### What is a "Carbon Footprint" in ML?

At its simplest, the carbon footprint of a machine learning model is the total amount of greenhouse gas (GHG) emissions generated throughout its entire lifecycle. These emissions are measured in **grams of CO2 equivalent (gCO2e)**, a unit that standardizes the warming effect of different gases (like methane) relative to carbon dioxide.

The ML lifecycle isn't just the `model.fit()` command. It’s a comprehensive pipeline:

1.  **Hardware Manufacturing (Embodied Carbon):** This is the "Scope 3" emission, often the most hidden. The GPUs, TPUs, CPUs, RAM, and servers in a data center don't appear from nowhere. They require mining rare minerals, complex manufacturing, and global shipping. This "embodied" carbon is a significant, fixed, upfront environmental cost before the hardware is even turned on. For GenAI, which relies on cutting-edge, massive-scale hardware, this footprint is substantial.

2.  **Data Collection, Storage, and Movement:** Large models require massive datasets. Collecting this data (e.g., scraping the web) costs energy. Storing it indefinitely on cloud storage costs energy. Moving a 50TB dataset from a storage bucket in one continent to a compute cluster in another consumes a significant amount of energy, not just in the compute but in the networking infrastructure along the way.

3.  **Model Training (Operational Carbon):** This is the most infamous source of emissions and a primary focus of this course. Training a large language model (LLM) or a diffusion model (GenAI) can involve thousands of high-performance accelerators (GPUs/TPUs) running at full power, 24/7, for weeks or even months. The energy consumed is colossal. The emissions from this step are a direct function of:

      * **Total Energy Consumed (kWh):** `(Power_per_chip * Num_chips * Time_in_hours) / 1000`
      * **Data Center Efficiency (PUE):** The energy for "overhead" (cooling, lighting).
      * **Grid Carbon Intensity (gCO2e/kWh):** The *source* of that energy.

4.  **Model Inference (Operational Carbon):** This is the "long tail" of emissions. While training happens once (or periodically), inference happens *every time* a user interacts with the model. For a product like Google Search or a GenAI chatbot with a billion users, the cumulative energy cost of inference can dwarf the one-time cost of training. Every prompt, every token generated, every image created costs a small amount of energy, which adds up at scale.

5.  **Model Retraining and Maintenance:** Models go stale. They need to be fine-tuned or completely retrained on new data, starting the training cycle's carbon cost all over again.

#### Why GenAI is a Special Case

This course is specifically for "GenAI Developers" because Generative AI has supercharged the scale of this problem.

  * **Parameter Scale:** Models have grown from millions of parameters (e.g., BERT) to hundreds of billions or even trillions (e.g., GPT-4, PaLM 2). There is a direct, near-linear relationship between the number of parameters and the computational (and thus, energy) cost to train.
  * **Training Data Scale:** GenAI models are trained on "internet-scale" data, often encompassing a significant portion of the accessible web, books, and image libraries.
  * **Inference Complexity:** Generating a 1000-word essay (autoregressive token generation) is far more computationally expensive than simply classifying an email as "spam" or "not spam."

Famous studies have tried to quantify this. A 2019 paper from UMass Amherst estimated that training a single large NLP model (Transformer) could emit as much carbon as five cars over their entire lifetimes. The models used in that paper are now considered small. The numbers for modern GenAI are significantly, almost unimaginably, larger.

#### Quantifying the Footprint: The Core Equation

In this module, you'll learn the practical formula that governs our work for the rest of the course. The carbon footprint of a compute job is determined by three main factors:

**Total Carbon = (Energy Consumed) × (Data Center Overhead) × (Grid Carbon Intensity)**

Let's break this down:

  * **Energy Consumed (kWh):** This is the energy drawn by the IT equipment (the GPUs/TPUs). As a developer, you influence this with:

      * **Hardware Choice:** Using a newer, more efficient chip (e.g., TPU v5) can reduce this.
      * **Model Architecture:** A smaller, more efficient model (e.g., a distilled model) will consume less energy.
      * **Code Efficiency:** Efficient data loading and processing, avoiding an "idle" GPU.

  * **Data Center Overhead (PUE):** This is the **Power Usage Effectiveness (PUE)**. A PUE of 2.0 means that for every 1 watt of energy the GPU uses, another 1 watt is used for cooling, lighting, and power conversion. A PUE of 1.0 is the "perfect" ideal. Google Cloud data centers are famously efficient, with an average PUE around 1.1, but this factor is still critical. As a developer, you *don't* control this directly, but you *choose* a provider (like Google) who does.

  * **Grid Carbon Intensity (gCO2e/kWh):** This is the single most volatile and, for this course, *most important* variable. It's the "dirtiness" of the electricity grid powering the data center. A data center in Norway (powered by hydro) might have an intensity of 10 gCO2e/kWh, while one in West Virginia (powered by coal) might be 800 gCO2e/kWh.

**This equation reveals the developer's new levers for change.** We've spent a decade optimizing "Energy Consumed" (Faster ML). This course is about optimizing the "Grid Carbon Intensity" (Greener ML). Running the *exact same* training job in a different *place* or at a different *time* can reduce its carbon footprint by 90% or more, with zero code changes. This module provides that critical "Aha!" moment, setting the stage for the solutions to come.

</details>

<details>
<summary><strong>Exploring Carbon Intensity on the Grid</strong></summary>

In this module, you move from theory to practice. Now that you understand *that* the "Grid Carbon Intensity" (gCO2e/kWh) is a critical variable, you'll learn how to *measure* and *query* it. This topic is a deep dive into the data, APIs, and real-world dynamics of the global electricity grid. You'll become a "grid-aware" developer, capable of seeing the world not just as a map of data centers, but as a living, breathing network of power generation.

#### The Grid is Not a Monolith

The first crucial concept is that "the grid" doesn't exist. Instead, the world is a patchwork of hundreds of different, often-isolated, regional grids. The carbon intensity of the grid in `us-east1` (Virginia) is completely independent of the grid in `us-west1` (Oregon).

This module introduces the two fundamental ways carbon intensity varies:

1.  **Spatial Variation (Varies by *Location*):** The *average* carbon intensity of a grid is determined by its **energy mix**—the breakdown of power sources.

      * **Clean Grids (Low gCO2e/kWh):**
          * **Hydro-dominated:** e.g., Quebec (`northamerica-northeast1`), Norway, Washington.
          * **Nuclear-dominated:** e.g., France (`europe-west9`), Ontario.
          * **Geothermal-dominated:** e.g., Iceland.
      * **Dirty Grids (High gCO2e/kWh):**
          * **Coal-dominated:** e.g., Poland (`europe-central2`), parts of the US Midwest, Australia.
      * **Mixed Grids (Variable gCO2e/kWh):**
          * **Natural Gas / Mixed Renewables:** e.g., California, Texas (`us-central1`), UK. These are often highly volatile.

2.  **Temporal Variation (Varies by *Time*):** This is the most dynamic and exciting part. The carbon intensity of a *single* grid changes every minute. Why? Because demand and supply are in a constant, delicate balance.

      * **Renewable Intermittency:** The wind doesn't blow 24/7, and the sun doesn't shine at night.
      * **The "Duck Curve":** You'll learn about this famous phenomenon, especially visible in solar-heavy grids like California. .
        1.  **Morning:** Demand ramps up, intensity is moderate.
        2.  **Mid-day:** Solar production is massive. It floods the grid with zero-carbon energy. Intensity plummets, and sometimes energy is so abundant it's "curtailed" (wasted). This is a *golden opportunity* to run compute jobs.
        3.  **Evening (The "Neck" of the Duck):** The sun sets *exactly* as people get home, turn on lights, and cook dinner. Demand spikes, but solar supply vanishes. To meet this "ramp," the grid must turn on "peaker plants," which are almost always fast-acting natural gas turbines. As a result, the carbon intensity of the grid can *triple* in just a few hours.
      * **Daily/Seasonal Cycles:** Nighttime often has lower demand, relying on "baseload" power (which could be clean nuclear or dirty coal). Spring might have more hydro (snowmelt), while winter has more wind.

#### The API: ElectricityMaps

To navigate this complex landscape, you need data. This course introduces the **ElectricityMaps API**, a key partner in this ecosystem. You will learn to use this API to query the state of the grid, just like you'd query a weather API.

You'll get hands-on experience (likely in a Python/Jupyter Notebook environment) making API calls to answer critical questions:

  * **`GET /v3/power-breakdown` (What is this grid made of?):**

      * You'll provide a `lat`/`lon` or a `zone` (e.g., `US-CAL-CISO`).
      * The API will return a JSON object: `{"power_production": {"wind": 4500, "solar": 12000, "hydro": 3000, "gas": 8000, ...}}`.
      * This allows you to see *why* a grid is clean or dirty *right now*.

  * **`GET /v3/carbon-intensity` (How clean is it right now?):**

      * This is the money shot. You'll query a zone and get back the current carbon intensity.
      * `{"zone": "US-CAL-CISO", "carbonIntensity": 120, "unit": "gCO2e/kWh"}`.

  * **`GET /v3/carbon-intensity/forecast` (How clean *will* it be?):**

      * This is the most powerful endpoint for *planning*.
      * The API provides a 24-hour (or more) forecast of carbon intensity.
      * `{"forecast": [{"dateTime": "...", "carbonIntensity": 110}, {"dateTime": "...", "carbonIntensity": 105}, ... ]}`.
      * This allows you to ask: "For my 4-hour training job, what is the *cleanest* 4-hour window in the next 24 hours?"

#### Average vs. Marginal Carbon Intensity

This module will also likely introduce a more advanced, critical concept: **Average vs. Marginal Intensity**.

  * **Average Intensity:** The metric we've discussed so far. It's the *average* gCO2e/kWh of *all* power on the grid. (Total CO2 / Total kWh).
  * **Marginal Intensity:** The carbon intensity of the *next* kWh of demand. This is arguably the *more important* metric. When you spin up a new training job, you are adding *new* demand. The grid must meet that new demand by turning *something* on. The "marginal" plant is often a natural gas peaker. So, even if the *average* intensity of the grid is low (because of lots of wind), the *marginal* intensity of *your specific job* could be very high, as you're the one "causing" the gas plant to fire up.

Understanding this difference is key to true carbon-aware computing. ElectricityMaps provides both "average" and "marginal" data. This module gives you the data literacy to understand these nuances and build truly intelligent systems. By the end, you'll be able to build a dashboard that ranks all of Google's data center regions by their *real-time* carbon intensity, setting the stage for Topic 3.

</details>

<details>
<summary><strong>Training Models in Low Carbon Regions</strong></summary>

This module is where you take your first major action. Armed with your knowledge of ML's carbon footprint (Topic 1) and your ability to query grid data (Topic 2), you will now implement the first and most powerful strategy for carbon reduction: **Spatial Shifting**.

The core idea is simple: **Run your workloads in a cleaner location.**

This module focuses on the "static" or "average" approach. This means making a long-term, strategic decision to default your training jobs to a region that is *consistently* low-carbon, based on its annual average grid mix.

#### The Static (Regional) Selection Strategy

As a developer, your default behavior is often to run code in the region closest to you (e.g., `us-east1` if you're in New York) or the region where your data lives. This is a decision made for convenience or latency, not for carbon.

This module teaches you to *change* that default. You will learn to:

1.  **Analyze Regions:** Use the data from Topic 2 (or Google-provided data) to analyze the *annual average* carbon intensity of all available Google Cloud regions.
2.  **Create a "Green List":** You'll identify the "best" regions for carbon. This list will almost certainly include:
      * `us-west1` (Oregon, USA - Hydro)
      * `northamerica-northeast1` (Montreal, Canada - Hydro)
      * `europe-north1` (Finland - Hydro, Nuclear, Wind)
      * `europe-west1` (Belgium - High renewable/nuclear mix)
3.  **Implement the Shift:** You'll learn the practical `gcloud` or API commands to execute your job in this new, cleaner region. Instead of the default:
    ```bash
    gcloud ai platform jobs submit training my_job ... --region us-east1
    ```
    You will *consciously* choose a cleaner alternative:
    ```bash
    gcloud ai platform jobs submit training my_job ... --region us-west1
    ```

This single change—running the *exact same code* in a different location—can reduce the emissions of that job by 80-95%. This is the lowest-hanging fruit for carbon-aware computing.

#### Google's Role: CFE and Regional Transparency

This isn't just about third-party data from ElectricityMaps. Google Cloud is a leader in this space and provides its own metrics. This module will introduce you to Google's **Carbon-Free Energy (CFE)** score.

  * **What is CFE?** Google's 2030 goal is to run on **24/7 Carbon-Free Energy**. This is a much higher bar than "100% renewable" (which is an *annual* match). 24/7 CFE means that for *every hour* of the day, at *every data center*, the energy consumed is *matched* by a carbon-free source on that *same local grid*.
  * **CFE Score:** Google publishes a CFE score (e.g., "97%") for each of its data center regions. This score tells you what percentage of the time that region is *actually* running on carbon-free energy.
  * **Actionable Data:** You'll learn to use this CFE score as a high-quality, Google-provided proxy for "cleanliness." When you see that `us-west1` has a CFE of 97% and `us-central1` has one of 50% (hypothetical numbers), the choice becomes obvious. Google's own tools, like the "Carbon-Free Energy for Google Cloud" dashboard, help you make this decision.

#### The All-Important Trade-Offs

This module would be incomplete if it pretended this decision was "free." Choosing a region has consequences, and a professional developer must weigh them. This is the most critical part of the topic.

1.  **Data Gravity & Egress Costs (The Big One):** This is the single biggest blocker. Your 50TB training dataset *lives* somewhere (e.g., a GCS bucket in `us-east1`). To train in `europe-north1`, you have two bad options:

      * **Move the Data:** This incurs massive **data egress fees**. Google charges you to move data *out* of a region and across continents. This cost can be *thousands of dollars*, potentially wiping out any financial savings from the compute. It also *costs* energy to move the data, creating its own (smaller) carbon footprint.
      * **Train Across the Wire:** Keep the data in `us-east1` and have the compute nodes in `europe-north1` read it remotely. This is a performance nightmare. The latency will kill your GPU utilization, making your job run 10x slower. This *longer runtime* might *increase* your total energy consumption, even on a cleaner grid\!
      * **Solution:** The *real* solution is to plan ahead. If you know you'll train in `europe-north1`, then *store* your data there in the first place. This is about *architectural* design.

2.  **Hardware Availability:** GenAI developers need the latest, most powerful hardware (e.g., TPU v5p, NVIDIA H100). What if the *cleanest* region (`us-west1`) only has older-generation TPUs, and the *newest* hardware is only available in a *dirtier* region (`us-east4`)?

      * This creates a complex optimization problem.
      * Finishing a job 2x as fast on an H100 (in a dirty region) might *still* use less *total energy* (Energy = Power x Time) than running 2x as long on a V100 (in a clean region).
      * This module teaches you to ask these questions and look at the trade-off between *chip efficiency* and *grid efficiency*.

3.  **Data Sovereignty and Residency:** You may be *legally* prevented from moving data. If you are working with European user data, the **GDPR** may require that data to *never* leave the EU. This *restricts* your choice of regions, but you can still pick the *cleanest* region *within* the EU (e.g., Finland over Germany).

4.  **Pricing:** Compute and storage costs are not uniform. The same VM or TPU-hour might be cheaper in one region than another. This must be factored into the total cost equation.

By the end of this module, you'll have the practical skills to select a region and submit a job, but more importantly, you'll have the senior-level engineering wisdom to understand the *full system* of trade-offs (Carbon, Cost, Latency, Law) that govern that decision.

</details>

<details>
<summary><strong>Using Real-Time Energy Data for Low-Carbon Training</strong></summary>

This module is the "GenAI" part of the course. It's where you graduate from a simple, static strategy (Topic 3) to a sophisticated, dynamic, and automated one. This is the cutting edge of carbon-aware computing.

The problem with the static "low-average" approach is that it misses massive opportunities. A region with high *average* wind (like Texas) might be clean *on average*, but if the wind stops blowing at 3 PM, its grid becomes very dirty, powered by natural gas. Conversely, a "dirty" grid might have moments of extreme "cleanliness" (e.g., a sunny, windy Sunday morning when industrial demand is low).

This module teaches you **Temporal Shifting**: running your job at the *cleanest possible time*. When combined with spatial shifting, this is the ultimate strategy: **Find the cleanest time in the cleanest available region.**

#### The "Carbon-Aware Scheduler"

This module is a capstone project. You will build, or at least design, a "Carbon-Aware Scheduler." This is a piece of automated logic (e.g., a Google Cloud Function) that orchestrates your training jobs intelligently.

Here is the architecture you will learn to build:

1.  **Job Ingestion:** A developer, instead of running `gcloud ai platform jobs submit...` directly, submits their job request to your *scheduler*. The request includes:

      * `job_script.py`
      * `dataset_location`
      * **Constraints:**
          * `deadline`: "I need this model by 9:00 AM tomorrow."
          * `hardware_required`: "I must use an A100 GPU."
          * `data_residency`: "Data must stay in the EU."

2.  **Carbon Data Query:** Your scheduler (which is a serverless function, e.g., a Cloud Run service) wakes up. It has a list of *viable* regions that satisfy the job's constraints (e.g., `europe-north1`, `europe-west4`, `europe-west9`, all in the EU and all have A100s).

      * It calls the **ElectricityMaps API** (from Topic 2).
      * It requests the `carbon-intensity/forecast` for the *next 24 hours* for *all viable regions*.

3.  **The "Greenest Slot" Algorithm:** This is the core logic. Your scheduler now has several time-series data streams. It must find the *optimal* time and place.

      * Assume the job is 8 hours long.
      * The scheduler "slides" this 8-hour window across the 24-hour forecasts for all viable regions.
      * It calculates the *average* gCO2e/kWh for each possible slot.
      * **Example:**
          * Slot A: `europe-north1` at 2:00 AM. Avg: 30 gCO2e/kWh
          * Slot B: `europe-west4` at 1:00 PM. Avg: 85 gCO2e/kWh
          * Slot C: `europe-north1` at 4:00 PM. Avg: 50 gCO2e/kWh
      * The algorithm finds the `min()`: Slot A is the winner. It is the "greenest" 8-hour window, within the deadline, that satisfies all constraints.

4.  **Job Execution:**

      * The scheduler doesn't run the job *now*. It uses **Google Cloud Scheduler** to create a *new* one-time trigger.
      * It sets the trigger for 2:00 AM.
      * The trigger's job is to call the Vertex AI API and *finally* submit the training job, with the correct parameters: `... --region europe-north1`.

This entire process is automated. The developer just says "run my job," and the system automatically finds the most environmentally friendly way to do it, balancing constraints and real-time grid data.

#### Advanced Concepts: Pause and Resume

What if your job is flexible? This module will likely explore even more advanced techniques, made possible by model checkpointing.

  * **The Problem:** A 12-hour job might start in a "clean" window (e.g., high solar), but 6 hours in, the sun sets, and the grid becomes 5x dirtier.
  * **The Solution:**
    1.  Your Carbon-Aware Scheduler monitors the *real-time* carbon intensity *while the job is running*.
    2.  It has a "dirtiness threshold" (e.g., 150 gCO2e/kWh).
    3.  If the grid intensity *crosses* this threshold, the scheduler *pauses* the training job. (e.g., `gcloud ai platform jobs cancel ...` but *after* the job has saved a checkpoint to GCS).
    4.  The scheduler then goes back to "monitoring" the forecast, looking for the *next* clean window.
    5.  When it finds one (e.g., 4:00 AM when wind picks up), it *resumes* the job from the last checkpoint.

This is "temporal shifting" at its most granular, "following the renewables" on an hourly basis. This requires a robust checkpointing strategy in your ML code (e.g., saving your model weights to GCS every 30 minutes), but it's the pinnacle of a carbon-aware workload.

#### Tying it to Google Cloud Tools

This isn't just theory. You'll learn to *implement* this using the GCP service stack:

  * **Cloud Run / Cloud Functions:** To *host* your scheduler logic.
  * **Google Cloud Scheduler:** To *trigger* the jobs at specific future times.
  * **Vertex AI:** The managed platform to *run* the ML training.
  * **Google Cloud Storage (GCS):** To store datasets and, crucially, model *checkpoints* to enable pause/resume.
  * **Artifact Registry:** To store your final model image.

This module is the true synthesis of the course. It combines ML engineering (checkpointing), data engineering (API-driven decisions), and cloud architecture (serverless orchestration) to build a system that is not just "green" by default, but *actively and intelligently* hunts for the cleanest possible energy to do its work.

</details>

<details>
<summary><strong>Understanding your Google Cloud Footprint</strong></summary>

This final module closes the loop. You've learned *why* ML has a footprint (Topic 1), *how to measure* the grid (Topic 2), and *how to act* on that data through spatial and temporal shifting (Topics 3 & 4). Now, you must answer the final question: **"Did it work?"**

This module is about **measurement, accountability, and reporting**. You can't optimize what you can't measure, and you can't claim success without data. This topic introduces the **Google Cloud Carbon Footprint** tool, the official "report card" for your environmental performance on GCP.

#### The Google Cloud Carbon Footprint Tool

This is a first-party tool available directly in the Google Cloud Console. It provides a comprehensive dashboard and data export capabilities to estimate the greenhouse gas (GHG) emissions associated with *your specific* usage of Google Cloud services.

You will learn how to use this tool to:

1.  **Visualize Your Footprint:** See your total emissions (in `kgCO2e`) over time (e.s., monthly).
2.  **Break Down by Project:** Identify which of your projects are the biggest emitters. Is it the "dev" project or the "prod-inference" project?
3.  **Break Down by Product:** See which *services* are responsible. Is your footprint dominated by **Compute Engine**, **Vertex AI**, or **BigQuery**?
4.  **Break Down by Region:** This is the most critical part for *validating* your work from this course. The tool will *show you* the emissions difference. You'll be able to see a chart with:
      * `us-east1`: 1500 kgCO2e
      * `us-west1`: 80 kgCO2e
        This provides direct, quantifiable proof that your regional selection (Topic 3) had a massive impact.

#### Understanding the Metrics: The Three Scopes

To use this tool professionally, you must understand the vocabulary of carbon accounting, specifically the GHG Protocol's "scopes":

  * **Scope 1:** *Direct* emissions. These are from sources Google *owns or controls*. For a data center, this is primarily the diesel fuel for its backup generators.
  * **Scope 2:** *Indirect* emissions from purchased energy. This is the **big one** for compute. It's the carbon footprint of the grid electricity that Google buys to power its data centers. *This is the number you have been working to reduce.*
  * **Scope 3:** *All other* indirect emissions. This is a massive, complex category that includes the embodied carbon of the hardware (manufacturing), employee travel, etc.

The Google Cloud Carbon Footprint tool *measures and reports* your share of Google's Scope 1, 2, and 3 emissions, pro-rated based on your usage.

#### The "Google is Carbon Neutral" Paradox

This module will address the most common question: "Wait, Google has been carbon neutral since 2007 and 100% renewable-matched since 2017. Why does this tool show *any* emissions at all?"

This is the most important concept in this module, and it's why this course exists.

  * **Annual Matching (What Google does):** Google's "100% renewable match" means that *over the course of a year*, they buy enough renewable energy (e.g., from a wind farm they built in Texas) to match 100% of their *total annual* electricity consumption. This is a *financial* and *annual* accounting mechanism.
  * **Hourly, Local Reality (What your code sees):** This does *not* mean that at 8:00 PM in Virginia (when the sun is down and wind is low), your training job is *physically* running on solar power. At that *moment*, your job is drawing power from the *local Virginia grid*, which is dirty (powered by natural gas or coal).
  * **The Tool's Honesty:** The Carbon Footprint tool *shows you the real, physical, location-based emissions (Scope 2)*. It shows you the carbon intensity of the *local grid* at the *time you ran your job*.
  * **Carbon Offsets:** Google's "carbon neutrality" (since 2007) is achieved by purchasing high-quality carbon offsets (e.g., funding a reforestation project) to "cancel out" these remaining emissions.

This is the crux: **Offsets are good, but *avoiding the emission in the first place* is far better.** This course (and Google's 24/7 CFE goal) is about moving from *offsetting* emissions to *not creating them*.

Your work in Topics 3 and 4 (spatial and temporal shifting) *directly* reduces the **gross Scope 2 emissions**, which is the *real* physical impact. The Carbon Footprint tool is your proof.

#### Closing the Loop: Reporting and Action

Finally, you'll learn what to *do* with this data.

  * **Exporting Data:** You can export your carbon data to **BigQuery**.
  * **Building Dashboards:** Once in BigQuery, you can connect it to **Looker Studio** to build custom dashboards for your team, your manager, or your company's ESG (Environmental, Social, Governance) department.
  * **Setting Carbon Budgets:** Just like financial budgets ("This project can't spend more than $1000/month"), you can set *carbon budgets* ("This project's goal is to stay under 500 kgCO2e/month").
  * **Justifying Your Work:** This data allows you to go to leadership and say, "By implementing the carbon-aware scheduler we learned in this course, our team reduced the physical carbon footprint of our AI division by 40% last quarter, with no impact on performance."

This module makes you not just a carbon-aware developer, but a carbon-accountable one, giving you the tools to measure your impact and drive institutional change.

</details>

-----

## Acknowledgement

This repository is for personal, educational use only, as part of the **"Carbon Aware Computing for GenAI Developers"** course.

This course is provided by **DeepLearning.AI** in collaboration with **Google Cloud**.

All course materials, lectures, and associated labs are the intellectual property of DeepLearning.AI and Google Cloud. All rights are reserved by them.