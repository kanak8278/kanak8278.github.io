---
title: "They'll Spend Anything"
date: 2026-08-01
categories:
  - blog
tags:
  - india
  - ai
  - semiconductors
  - research
  - opinion
layout: blog-post
toc: true
toc_label: "Contents"
read_time: true
related: false
classes:
  - content-page
  - blog-post-page
  - wide-figures
  - tex2jax_ignore
excerpt: "Except on anything ambitious. A rant about India's richest families, with receipts — a first chip fab opening at 90nm, frontier models that cost less than a soft-drink launch, and a cricket team that sold for twice everything Indian rocketry has ever raised."
header:
  teaser: /assets/images/posts/theyll-spend-anything/04c-ad-vs-deepseek.png
---

<div class="post-intro" markdown="1">

**Except on anything ambitious.** A rant about India's richest families, with receipts.[^scope]

I'm angry and I'm partial. Every number here is sourced, so you can check whether the anger is earned — working notes, arithmetic and exchange rates are on [the sources and calculations page](/blog/theyll-spend-anything-sources/).

**One thing before the numbers, because it matters.** I have nothing against the Ambanis, and this is not a piece about one family. Reliance has built things this country genuinely needed. Jio changed how a billion people use the internet. The green-energy build at Jamnagar — solar, storage, electrolysers — is serious, capital-heavy industrial work that most conglomerates in the world wouldn't attempt, and I'd rather they were doing it than not.

I lean on them because they are the largest and the best-documented, so the figures are in public filings rather than in my imagination. But the question underneath all of this is colder than any one family: **what does it actually cost to build state-of-the-art technology, and do we have the capability to do it?** The Ambani numbers are how I make that cost legible. They are the measuring stick, not the target.

If you think I'm being unfair, skip to [Where I might be wrong](#where-i-might-be-wrong) at the bottom. I got there first.

</div>

India's first real chip fab will start commercial production in mid-2028, mostly at 90 nanometres.

The world shipped 90nm in 2004. The iPhone did not exist yet.

Tata told its own shareholders the Dholera fab would start at 28nm. It's starting two nodes below that, three years later than announced. And 28nm was itself a 2011 node. So the optimistic version of India's chip debut was fifteen years behind, and we missed it.

![The node gap](/assets/images/posts/theyll-spend-anything/01-node-gap.png)

And that is the fab that is actually getting built. The other one — Adani's $10 billion venture with Israel's Tower Semiconductor — has been in "strategic pause" since January 2026 over subsidy terms. The headline number from last year is currently zero.

That's the state of Indian silicon. Before the statistics, let me show you what the alternative looks like.

## The richest man in the world

I don't hold Elon Musk up as an idol. I disliked most of what he did at DOGE and I have no interest in defending him as a person.

Watch what he did with money, though, because it's the cleanest version of the argument I'm about to make.

He walked away from PayPal in 2002 with **$180 million.** He put **$100 million into SpaceX, $70 million into Tesla, $10 million into SolarCity** — all of it, into rockets and electric cars, at a time when both were considered a rich man's way of becoming a poor one.

By late 2008 it had nearly worked exactly that way. Tesla had **$9 million** in the bank and was days from running out. SpaceX had failed three launches in a row. Musk was borrowing money from friends to cover his own rent. Then the fourth Falcon 1 reached orbit, NASA signed a **$1.6 billion** resupply contract, and both companies survived by a margin measured in weeks.

He is now worth roughly **$833 billion**, and briefly became the first person in history past a trillion when SpaceX listed in June 2026 at a $1.77 trillion valuation.[^musk]

Notice the order, because it's the same order as everything else in this piece. He was not rich and therefore able to take the risk. **He took the risk and that is where the money came from.**

Now ask what our richest man's group is putting its next big cheque into. A ₹30,000 crore chain of factories making biscuits and fizzy drinks. I'll come back to that number, because it turns out to be almost exactly what a frontier AI lab would cost.

That gap is not a wealth gap. India has the wealth. It's an appetite gap — and it shows up in the national numbers.

## The number

India spends **0.64% of GDP on R&D.**[^rdgdp] Korea spends 4.91%. The US 3.48%. China 2.43%.

Everyone quotes that number and blames Delhi. Look at the second one instead — who actually pays for the research.

![R&D: how much, and who pays](/assets/images/posts/theyll-spend-anything/02-rd-who-pays.png)

In every serious country, private money does three-quarters of it. Here it does 41%. The government is not the bottleneck. The government is the only one showing up.

And before someone says "but our companies are smaller" — it's a ratio, not a size. Reliance's own audited filing puts FY25 R&D at **₹4,185 crore**, which is **0.75% of turnover.**[^rilrd] Huawei spends **20.8%.**[^huawei] Samsung spends 11.3%.

Twenty-eight times the intensity. Not twenty-eight percent more. Twenty-eight times.

Two things make it worse. Of that ₹4,185 crore, ₹2,652 crore is capital — buildings and kit. The recurring research line, the one that actually maps to what Huawei reports, is ₹1,533 crore. That's 0.27%. And Reliance's group revenue, roughly $122 billion, is *larger* than Huawei's $119 billion. So the small-company defence isn't available either.

![R&D as a share of revenue](/assets/images/posts/theyll-spend-anything/03-rd-intensity.png)

## "Let the big boys in the Valley do it"

That's Nandan Nilekani, on stage at Meta's summit in 2024, sitting next to Yann LeCun:

> "Let the big boys in Silicon Valley do it, spending billions of dollars. We will use it to create synthetic data, build small language models quickly."

He doubled down in December: *"Foundation models are not the best use of your money."*

Two problems with it, though.

**The first is that it eats itself.** Who uses a model that isn't good? Nobody. And if nobody uses it, where does the usage data come from — the data you were going to train your small models on, the data that was supposed to be India's whole advantage? The plan needs a flywheel. A flywheel needs something worth using at the centre of it. You cannot collect exhaust from a car that never starts.

There's a nastier version of this. Google Research India's Manish Gupta made it politely; I'll make it rudely. Nilekani built Aadhaar by building the layer underneath. He didn't wait for someone in California to ship an identity stack and then do use cases on top of it. He's prescribing for the country a strategy he personally refused to follow.

And what does the strategy actually get us, if it works perfectly? We become very good at wrapping other people's models. We catch up with the West, and now with China, permanently one release behind, forever. That's not a strategy. That's a description of the last thirty years with the nouns updated.

**The second problem is the price.** The argument was built on a number that no longer exists.

## Training is not the bottleneck any more

DeepSeek trained V3 for **$5.58 million** of compute.[^deepseek] Peer-reviewed, published in *Nature*. R1's reinforcement learning run was $294,000.

Yes, that's the compute bill, not the company. The GPUs came earlier: High-Flyer, a Chinese quant hedge fund, bought 10,000 A100s for about $147 million in 2021 — before the export controls, with its own trading profits, because VCs wouldn't fund Liang Wenfeng.

A hedge fund. Not a conglomerate. Not a state. A guy who made money trading stocks decided to spend $147 million on GPUs and then handed the results to the world for free.

Here's the part that should sting.

When Liang bought those GPUs in 2021 he was a quant fund manager nobody outside Chinese trading circles had heard of. Not on any rich list. As recently as early 2026, Hurun still had him at $4.6 billion — respectable, and a small fraction of Mukesh Ambani's.

Today he is worth **$36 billion.** Richest AI-model founder on earth, eighth-richest man in China.

Drop him onto India's rich list and he lands **third**[^liang] — level with Roshni Nadar Malhotra, behind only Ambani and Adani. Above every other name in this piece.

![Where Liang Wenfeng would rank in India](/assets/images/posts/theyll-spend-anything/04b-liang-ranking.png)

Read the order carefully, because it's the whole argument. He did not get rich and then take the shot. He took the shot with a fraction of their money, and the shot is what made him rich.

Reliance's beverage capex is ₹8,000 crore.[^campa] About **$910 million.** That is 163 DeepSeek-V3 training runs. Or six High-Flyer GPU clusters. Spent on fizzy drinks.

![What a frontier model actually costs](/assets/images/posts/theyll-spend-anything/04-training-costs.png)

And it isn't only Reliance. Once you have $5.58 million as a unit of measurement, corporate India starts looking insane.

Hindustan Unilever spent **₹6,028 crore** on advertising in FY25. $685 million. That is 123 DeepSeek V3 training runs, to sell soap.[^ads]

Indian brands spent **₹4,000 crore** in a single year on ads placed *inside* Blinkit, Zepto and Instamart — not building the apps, buying banner space in them. Eighty-one training runs.

Then there's the one that actually made me put my laptop down. Add up everything DeepSeek spent to build V3 and R1 — all 10,000 GPUs, both training runs, the entire thing — and you get about **$153 million.**

ITC's advertising and promotion budget for FY25 was **$151 million.**

A company that sells cigarettes and biscuits spends, on advertising, in one year, what it cost to build the model that knocked **$589 billion off Nvidia in a single day** — the largest one-day loss in the history of markets.

![Everything DeepSeek spent, priced in Indian advertising](/assets/images/posts/theyll-spend-anything/04c-ad-vs-deepseek.png)

I'm not claiming ₹8,000 crore of cola money converts one-to-one into a frontier lab. It doesn't — you need people, data, and years, and both bets can be rational at once. I'm claiming the sentence "foundation models are too expensive for India" stopped being true somewhere around 2024, and our national AI doctrine is still built on it.

## What China's private money did while we were arguing

Alibaba put **380 billion yuan (~$53 billion)** into AI and cloud over three years. More than it spent on that entire category in the previous decade.

They also open-sourced 400+ Qwen models. Over a billion downloads on Hugging Face. Qwen has overtaken Llama as the world's default open model.

Now line up the open-weight models.[^params]

![Open-weight model scale](/assets/images/posts/theyll-spend-anything/05-open-model-scale.png)

Someone will say China's edge is just state money. It isn't only that, and the numbers say so. Beijing's flagship chip vehicle, Big Fund III, is **$47.5 billion** — about the size of the entire US CHIPS Act. Alibaba's commitment, from one company, is **larger than that.**[^bigfund]

That's the part worth sitting with. In China both kinds of capital show up, and the private cheque is the bigger one. Here only the state shows up, and it's the side with the smaller balance sheet.

Set all of that against India's share of global AI funding — every company, every round, all of it: **0.6%.**

![India's share of global AI funding](/assets/images/posts/theyll-spend-anything/06-global-ai-share.png)

## Our one champion runs on government GPUs

Sarvam raised $234 million in June 2026 at a $1.5 billion valuation, HCLTech leading with $150 million. That's real and I'm glad it happened. Roshni Nadar Malhotra wrote that cheque and she is, as far as I can tell, the only heir in this country doing anything at the frontier.

Sarvam-105B was trained on **1,000+ H100s from the government's Yotta cluster**, under IndiaAI's 4,000-GPU allocation. Our flagship private AI company trains on state compute.

And here is the easy version of everything I am asking for.

They don't have to build anything. Nobody has to hire a research org from scratch, or buy a fab, or learn what a transformer is. Sarvam already exists. It has the people, it has models shipping, it just doesn't have money at the scale the work needs — which is why it's training on the government's GPUs.

Sarvam's entire Series B is **$300 million** — and that's the full round; the June first close was $234 million. Mukesh Ambani could put in triple the whole round and not feel it leave. A **$1 billion** cheque into Sarvam is **0.9% of his net worth**, and it would make Sarvam one of the best-funded labs outside America and China overnight.

That's the whole ask. Fund the thing that already works. Let them buy their own GPUs.

And it wouldn't stop there, which is the actual point. Roshni Nadar wrote one $150 million cheque and made Sarvam a unicorn in an afternoon. One more of those from a bigger balance sheet and funding frontier research becomes a normal thing that Indian business houses do — and the next twenty-six-year-old with a good idea doesn't have to fly to Khosla Ventures to get started.

## The guy who actually tried, and quit

Bhavish Aggarwal announced ₹2,000 crore into Krutrim in February 2025, promised ₹10,000 crore more, and started designing chips.

By May 2026, Krutrim had **paused chip design and foundation-model work** and pivoted to selling cloud.

I want to say this clearly: he tried. He tried harder than anyone else on that rich list, and he ate the loss publicly. The lesson everyone else took from it is the wrong one.

## The landlord bet

Ambani: **$110 billion** over seven years for AI data centres in Jamnagar. Adani: **$100 billion**. Government target: $200 billion.

Enormous. Real. And it's a landlord business.

Jamnagar runs on Nvidia Blackwell. IndiaAI's 38,000+ GPUs are H100s and H200s. Microsoft has pledged $17.5 billion and Google $15 billion into Indian AI infrastructure — foreign capital, foreign silicon, Indian land and power.

It's the GCC playbook with racks instead of ticket queues. It comes here because electricity and land are cheap, same reason a back office opens in Bangalore instead of Boston. Nobody at this scale has stood up a program aimed at building a model that beats Mistral, let alone GPT.

We're going to be the world's best-equipped hosts. Hosting is a good business. It is not the same business.

![$242 billion to host other people's AI, $639 million to build our own](/assets/images/posts/theyll-spend-anything/07-landlord-vs-builder.png)

## The excuse I keep hearing, and Korea's answer to it

"Asian family businesses don't do moonshots. That's an American thing."

February 8, 1983. Tokyo. Lee Byung-chul, 73 years old and dying of lung cancer, announced Samsung would put roughly **$400 million** — the group's reserves — into semiconductors.

His entire senior team opposed it. Intel called it delusional. Mitsubishi published a report listing five reasons Samsung would fail.

First 1Mb DRAM in 1986. By 1992 Samsung passed Toshiba to become the world's largest DRAM maker. Lee died in 1987 and never saw it work.

Korea's per-capita income in 1953 was below Somalia's. Today two Korean companies make two-thirds of the memory in every device on earth.

So no. It isn't cultural. A dying man with a hostile board did it in a poorer country forty-three years ago.

## Nobody wants to come home, and it isn't about salary

July 14, 2026: the Department of Space told ISRO centres to **stop approving resignations** from scientists on Gaganyaan and other strategic missions. Every exit now routes through Delhi.

Between 100 and 120 had already left. Nearly 80 from the UR Rao Satellite Centre alone. No pay revision, no restructuring, no statement about why people were walking. Just a heavier door.

But I want to go past the easy joke about that memo, because the deeper problem isn't the memo and it isn't even the money.

**Only 26.6% of Indian researchers who go abroad ever come back.** Ask any of them why and you don't get "the salary." You get procurement rules. Multi-layered file approvals. Tendering delays. Rigid hierarchy and no autonomy. Everyone in the diaspora has heard the stories — the professor who waited fourteen months for a GPU purchase to clear, the lab where you report to someone who has never read your field.

So a serious AI researcher at DeepMind or Meta does the maths and it isn't close. Even if Delhi tripled every salary tomorrow, they still wouldn't come, because what they'd be buying is a decade of paperwork.

Now here is the part that makes me genuinely angry, because it's the one thing that was fixable.

**A privately funded lab has none of those problems.** No tendering. No file approvals. No secretary in Delhi signing off on a compute purchase. If one Indian business house had stood up a properly funded research institute — real money, real autonomy, insulated from the ministry — that is precisely the pitch that pulls people home. Not patriotism. Not a salary bump. *"Come and work on the hard thing, and nobody will make you fill in a form to do it."*

That offer has never been made. Not once, by anyone on the rich list. So our people stay where the work is, and we tell ourselves it's a brain drain problem, as if brains drain by themselves.

### Two weeks ago we were all very proud

On 18 July 2026, Skyroot's Vikram-1 reached orbit from Sriharikota — the first privately built Indian rocket to do it, making India the third country in the world with private orbital launch capability, after the United States and China. Everyone posted about it. I posted about it.

Now look at who paid for it. Skyroot has raised **$150 million** across its entire life. The money came from Sherpalo Ventures, Singapore's **GIC**, Singapore's **Temasek**, and **BlackRock**.

Two Singaporean state investment funds and an American asset manager put an Indian rocket into orbit.

Not one Indian conglomerate. Not one. A pension fund in Singapore has more conviction in Indian rocketry than the people whose own engineers built it.

## Cricket

I support Mumbai Indians. Please read the next four numbers in that spirit.

- Reliance's Viacom18 paid **₹23,758 crore (~$2.9 billion)** for IPL digital rights, 2023–27.
- MI's enterprise value: **$2.2 billion.**[^mi]
- Running MI for one year, FY24: revenue ₹737 crore, profit ₹109 crore, so about **₹628 crore (~$71 million)** in costs.
- Everything Indian spacetech has ever raised: **$871 million.**

Reliance paid three times our entire private space industry's lifetime funding for the right to *stream a cricket tournament.*

![Cricket money vs deep-tech money](/assets/images/posts/theyll-spend-anything/08-cricket-vs-deeptech.png)

And Anant Ambani's wedding: estimates run **$150 million to $600 million** across four months of events. At the figure everyone quotes, that is about **70% of every rupee India's private rocket industry has raised in its entire existence.** Another fortnight of party and you'd have bought the whole sector outright.[^wedding]

It's his money and his son and I genuinely don't care how anybody celebrates a wedding. What I'm pointing at is the divide. In the same country, in the same decade, one family can spend more on a party than an entire industry of engineers can raise to build rockets — and the engineers have to go to Singapore to get funded. That gap isn't a coincidence or bad luck. It is the same decision, made over and over, about what deserves money here.

## The next generation

Leave the fathers aside. Their children inherited the money *and* the choice of what to point it at. Here is the roster.

- **Akash Ambani** — chairs Reliance Jio, runs Jio Platforms. Telecom and distribution, over $100 billion of value. A real job, well done.
- **Isha Ambani** — Reliance Retail, $30 billion of revenue, plus Reliance Consumer Products. At the 2026 AGM she set RCPL's target at **₹1 lakh crore of revenue by 2030**, backed by **₹30,000 crore — about $3.4 billion — over three years** in food parks making beverages, chocolates, biscuits and packaged staples.
- **Anant Ambani** — Reliance New Energy, and Vantara: 3,000 acres, $150–300 million of capital, ₹150–200 crore a year to run. Genuinely good conservation work.
- **Karan Adani** — Managing Director, Adani Ports.
- **Jeet Adani** — Adani Airports, and he does hold the AI, data centre, defence and copper briefs. The closest any heir gets to the frontier. It is still the landlord bet: racks, not models.
- **Aryaman Vikram Birla** — Chairman of Royal Challengers Bengaluru, after an Aditya Birla-led consortium bought the franchise in March 2026 for **$1.78 billion**, the most expensive team sale in IPL history. His sister **Ananya Birla** runs Svatantra Microfin, plus a music career and beauty and lifestyle brands.
- **Parth Jindal** — JSW Cement, JSW Paints, half of Delhi Capitals, Bengaluru FC.
- **Kavin Bharti Mittal** — Hike, which was a WhatsApp rival that reached 100 million users and a $1.4 billion valuation in 2016. He shut the messenger in 2021 and turned it into Rush, a real-money gaming platform, then shut all of it in September 2025 after the gaming ban. $261 million raised. Four million left on the balance sheet.
- **Roshni Nadar Malhotra** — HCLTech, and the $150 million into Sarvam. The exception.

Read the list again as a list. Cement, paint, ports, airports, retail, biscuits, microfinance, a pop career, and **two IPL franchises.** One AI cheque, from one person.

Look at what those businesses have in common and you find the whole thesis of this piece. Cement, refining, petrochemicals, copper, ports, airports — commodities and tollbooths. Retail, telecom, FMCG — distribution and sugar. Cricket — a media asset with a proven audience and a contracted revenue line.

Every single one is a business where somebody else has already demonstrated that the money is there. You are buying a known payoff at a negotiated price. There is no version of a cement plant where you spend four years and find out the thing doesn't work.

That's the actual pattern. Not stinginess — they spend colossally. Our richest families are commodity traders and shopkeepers at scale, extremely good at both, and they will deploy tens of billions the moment the return is arithmetic rather than a bet. Ask them to put a tenth of that into something where the honest answer is *we don't know if this will work*, and the money evaporates.

And that RCB transaction deserves its own sentence, because it happened five months ago and nobody framed it this way: **one Indian business family spent $1.78 billion on a cricket team — more than twice everything India's private space industry has raised since it was legally allowed to exist.**

I'll take the wildlife sanctuary over most things billionaires' children do, and running Jio is genuinely hard work. My problem isn't that any single item on that list is bad. It's the shape of the whole list.

![What the heirs built vs what India's AI companies raised](/assets/images/posts/theyll-spend-anything/09-heirs.png)

## For scale

The Trump administration froze about **$2.6 billion** at Harvard in one political fight.

India's FY26 budget: all 23 IITs, **₹11,349 crore (~$1.29 billion)**. IISc, our single best research university, **₹845 crore (~$96 million)** — a cut from last year.

One American university's funding tantrum was, in dollar terms, bigger than India's entire elite science and engineering system for a year.[^scale]

![Harvard's funding fight vs India's elite science budget](/assets/images/posts/theyll-spend-anything/10-harvard-vs-iits.png)

## So what would it actually cost

Being honest about which of these is cheap and which isn't:

**A frontier AI lab — cheap.** 500 top Indian-origin researchers brought home at a blended $1.2M each is $600M/year. Ten thousand H100-class GPUs, roughly $300M. Data, power, ops, $200M. **About $1.1 billion for year one, ~$3 billion over three years.**[^lab]

Now hold that $3 billion next to one number, announced by Reliance this June.

Isha Ambani told the 2026 AGM that Reliance Consumer Products will put **₹30,000 crore — about $3.4 billion — into food parks over the next three years.** Highly automated, AI-enabled factories, for making beverages, chocolates, biscuits and packaged staples.

Three years of biscuit and fizzy-drink factories: **$3.4 billion.** Three years of a world-class frontier AI lab: **$3.0 billion.**

They are not choosing between these because they can't afford both. They already picked, and they picked the biscuits, and the biscuits cost more.

For the rest of the scale: $3 billion is **2.7% of the $110 billion data-centre pledge Reliance has already announced** — not 2.7% of their wealth, 2.7% of a cheque they have already committed to writing. It's roughly 3% of Mukesh Ambani's personal fortune. It's three and a half times everything Indian spacetech has ever raised.

And over those same three years, Zepto will burn about **$1.5 billion** getting groceries to people ten minutes faster. The lab costs twice that. The difference is that at the end of three years you'd own a frontier model instead of a fleet of scooters.

**A sub-5nm fab — genuinely expensive.** Samsung's Taylor fab went $17B → $25B. TSMC Arizona went $12B → $165B → $265B. A first-of-its-kind leading-edge fab in India would realistically be **$25–40 billion**, and India gets no discount on the part that matters: 70–80% of a fab is ASML and Applied Materials tooling, priced in dollars, the same in Dholera as in Phoenix. Plus EUV is export-controlled, so it may not be buyable at any price.

So I'll concede the fab. That one is hard, slow, and possibly blocked.

The model lab is not hard. It's about 3% of one man's fortune, and less than he's already putting into biscuit factories. It is a decision.

![What it would cost, against what's already being spent](/assets/images/posts/theyll-spend-anything/11-what-it-would-cost.png)

And we are good at this when we decide to be. Chandrayaan-3 landed for $75 million. Mangalyaan reached Mars on the first attempt for $74 million, less than the film *Gravity* cost to make. Both were ISRO — government money, government salaries, and over a hundred of those engineers walked out this past year. None of it came from Indian private capital.

## Willful blindness

Stanford's Global AI Vibrancy ranking puts India third in the world, behind only the US and China, up from seventh. That line has been in every ministerial speech and every LinkedIn post for a year.

Nobody quotes the scores. US 78.60. China 36.95. India **21.59.**

Third place, at 27% of the leader. It's a podium finish in the way that finishing a marathon an hour behind is a podium finish, if only three people entered.

Here is the same country, same year:

- 0.6% of global AI funding
- Our largest open model is outweighed 27 to 1 by one Chinese startup founded in 2023
- First fab opening at 90nm in 2028
- The best-funded private attempt at chips and models, Krutrim, shut both down and went into cloud resale
- Over a hundred ISRO scientists gone, and the official response was to make resigning harder
- Our largest company spends 0.75% of turnover on research

Third in the world.

I'm not saying the ranking is fabricated — it measures talent, papers, startup counts, and on those we genuinely are strong. That's exactly what makes it dangerous. It's a real number that lets you skip the question of what we actually *built*.

This is the part I find hardest to write without swearing. We are not ignorant. We are ignorant **by choice.** A politician says we're a superpower, and we repeat it, and we repeat it, and somewhere in the repetition it stops being an aspiration and becomes an excuse not to check. Every one of the numbers above is public. Nobody had to leak anything to me. I found them in annual reports and budget documents in an afternoon.

Open your eyes and we are not leading. We are not close to leading. We are being told a story, and we like it, so we don't check.

Yes, we did things. UPI is genuinely world-class. The 5G rollout changed how a billion people use the internet. Both are real and I'm proud of both. Both are also *deployment*. Neither required us to invent something nobody had proven would work. That's the muscle we haven't used.

## The question

Steve Jobs to John Sculley: *do you want to sell sugar water for the rest of your life, or come with me and change the world?*

I used to think our answer was "they won't spend the money." That's wrong now. They'll spend $110 billion on data centres, ₹30,000 crore on biscuit factories, $2.9 billion on cricket streaming, $1.78 billion on a cricket team, $600 million on a wedding.

They'll spend anything. They just won't spend it on something that might not work.

Sculley's answer, when Jobs asked him, was to leave the sugar water. Ours have looked at the same question and decided sugar water is a very good business — and they're right, it is. Campa is India's fourth-largest soft drink in four years. That's real commercial skill. It's just that the entire ambition of the richest people in a country of 1.4 billion turns out to be selling more units of a thing that already exists, to people who already exist, at a margin somebody already proved.

Not greed. Not stupidity. Certainly not a shortage of engineers. Just an unbroken preference for the sure thing, held by everyone with enough money to do otherwise. I'm risk-averse too, so I understand it. The difference is that I don't have $111 billion, and nothing is lost when I flinch.

Fifty years from now nobody is going to remember who owned Campa Cola.

Yeah, it's a rant. I don't think many people will read it.

---

## Where I might be wrong

Everything here is checkable. Go and check it. The numbered marks through the piece link down to here — each one is a place the argument could be attacked, and my honest answer.

[^scope]: **I focus on Ambani and Adani** because they're the largest and the best-documented, not because they're uniquely guilty. Someone should run these same numbers on Mittal, Birla, Jindal and Mahindra. I'd read it.

[^rdgdp]: **This number may be better than I've said.** I used the [Economic Survey 2025-26](https://www.businesstoday.in/economic-survey/story/indias-rd-spend-at-06-of-gdp-due-to-low-contribution-from-private-sector-economic-survey-513453-2026-01-29) figures — 0.64% of GDP, 41% private — because the four-country comparison comes from that same table. But a [July 2026 parliamentary answer](https://www.business-standard.com/economy/news/r-d-spending-in-india-only-0-84-of-gdp-in-fy24-govt-tells-parliament-126073001491_1.html) put FY24 at 0.84%, with private spend passing government spend for the first time. That's genuinely good news, and it's still about a third of China's intensity.

[^rilrd]: **This ratio is standalone, not group.** ₹4,185 crore is the audited Companies Act disclosure for RIL the company, in the Technology Absorption annexure of the [Integrated Annual Report 2024-25](https://www.ril.com/reports/RIL-Integrated-Annual-Report-2024-25.pdf). Whatever Jio and Reliance Retail spend on research isn't in it and isn't disclosed anywhere, so the true group figure is higher than 0.75% and nobody outside Reliance knows by how much. I had this wrong in an earlier draft: I used 0.34%, from a third-party estimate divided by *group* revenue, which understated them by more than 2×.

[^huawei]: **Comparing Reliance to Huawei is not like-for-like.** [Huawei](https://www.huawei.com/en/annual-report/2024) is a pure technology company; Reliance is mostly refining and retail, which are less R&D-intensive everywhere on earth. Fair objection. It's also the point — choosing which business to be in is the decision I'm complaining about.

[^deepseek]: **This is the pre-training compute run only.** It excludes the GPUs, the salaries and every failed run. [*The Register* made exactly this criticism](https://www.theregister.com/2025/09/19/deepseek_cost_train/) and was right to. Anyone quoting $5.58M as "the cost of a frontier model" — including me, if I get sloppy — is misleading you. The honest version is $147M of hardware plus single-digit millions per run. Both figures come from [DeepSeek's own peer-reviewed paper in *Nature*](https://www.cnn.com/2025/09/19/business/deepseek-ai-training-cost-china-intl).

[^ads]: **Advertising money is not spare money.** HUL's ₹6,028 crore is *how HUL sells soap*; cut it to zero and you get no soap sales and still no frontier model. I use training runs as a unit of measurement because $5.58 million is a figure nobody can place — not as a claim that ad budgets convert into labs. [Ad data via Storyboard18 and the Pitch Madison Advertising Report 2026.](https://www.medianews4u.com/indias-ad-market-crosses-%E2%82%B91-55-lakh-crore-in-2025-digital-now-60-of-adex-madison-report-2026/)

[^params]: **Parameter count is a weak proxy.** Kimi K3 being 27× Sarvam-105B doesn't make it 27× better — a well-trained small model beats a big one all the time, and Sarvam-105B is a real from-scratch multilingual model I'd much rather exist than not. What the ratio measures is how much compute someone was willing to point at the problem.

[^lab]: **This is the only invented number in the piece** — my estimate, not a company figure. Built from [published Meta research-scientist comp bands](https://www.levels.fyi/companies/meta/salaries/research-scientist) ($305K–$581K) and street GPU prices: 500 researchers at a blended $1.2M = $600M/yr, ~10,000 H100-class GPUs ≈ $300M, data and ops ≈ $200M. The compute line could be 2× off in either direction. The talent line assumes you can actually persuade 500 such people to move to India, which is a question money alone doesn't answer — and arguably the real constraint.

*Full sourcing, exchange rates, and arithmetic: [the sources and calculations page](/blog/theyll-spend-anything-sources/).*

[^campa]: **₹8,000 crore is the top of a range, and it isn't only Campa.** Reliance Consumer Products' announced beverage capex is [₹6,000–8,000 crore to March 2027](https://www.business-standard.com/companies/news/reliance-consumer-products-beverage-expansion-coca-cola-pepsi-campa-125061900145_1.html), covering Campa *and* Independence and other RCPL brands, not Campa alone. I use the upper bound. At the lower bound the multiple is 122 training runs rather than 163.

[^liang]: **Different index, different date — treat the placement as indicative.** The Indian figures are the Hurun Rich List 2026 (March 2026, converted at ₹88/\$); Liang's \$36.0bn is the Bloomberg Billionaires Index (July 2026), after DeepSeek's \$7.4bn round at a \$50bn valuation and assuming his ~78% holding. At \$36.0bn against Roshni Nadar Malhotra's \$36.4bn he is level with her, not clearly ahead — "third" here means the same bracket, not a verified league position.

[^wedding]: **The fortnight is rhetorical rounding; the ratio is the real claim.** At the widely quoted \$600m upper estimate the wedding is 68.9% of the \$871m Indian private spacetech has raised — a gap of \$271m. The events ran across roughly four months in 2024, so "another fortnight" is a figure of speech for what that remaining \$271m would represent, not a costed figure. Estimates run \$150m–\$600m; at the low end the comparison is 17% and I wouldn't make the point at all.

[^mi]: **Enterprise value, not brand value — two different numbers.** The Houlihan Lokey IPL Valuation Study 2025 puts Mumbai Indians' *institutional enterprise value* at \$2.2bn and its *standalone brand value* at \$242m. The larger figure is the business; the smaller is the brand alone. Both are in the same study.

[^scale]: **Straight exchange-rate conversion, no PPP adjustment — and PPP would flip this one.** Rupee figures converted at ₹88/\$. At World Bank purchasing-power rates (roughly ₹23 to the dollar) the IIT + IISc + IISER allocation is worth about **\$6 billion** of local buying power, which is more than double the Harvard figure, not smaller than it. So this is a nominal-dollar comparison and I'm not going to pretend otherwise. What doesn't get a PPP discount is the part of frontier science that's bought on world markets: instruments, GPUs, cleanroom tooling, journal access, and above all researchers, who are recruited against Zurich and Seattle salaries rather than local ones. That's why the nominal number still tells you something. But if you want to argue India's elite institutions are better funded in real terms than this section implies, you're right, and the honest version of my complaint is the private-sector one that the rest of the piece is about.

[^musk]: Musk's net worth moves violently and these figures are a snapshot: Bloomberg Billionaires Index had him at roughly \$833bn in mid-July 2026, up around \$214bn year to date, after peaking above \$1 trillion on the SpaceX IPO of 12 June 2026 (\$1.77 trillion, the largest listing in history) and falling back as the stock corrected. The PayPal allocation — \$100m SpaceX, \$70m Tesla, \$10m SolarCity out of \$180m — is Musk's own account, repeated consistently over the years. The 2008 near-death detail (Tesla down to \$9m, three consecutive Falcon 1 failures, the \$1.6bn NASA CRS contract) is documented across CBS News and Musk's own interviews. I'm using him as an illustration of risk appetite, not endorsing his politics or his management.

[^bigfund]: **Not strictly like-for-like, and worth saying.** Alibaba's ¥380bn (~\$53bn) is a three-year capital-expenditure commitment on AI and cloud infrastructure. Big Fund III's \$47.5bn is the registered capital of a state investment vehicle that takes equity stakes across the chip supply chain — a different instrument on a different clock, and chips are not the same industry as cloud. I'm comparing the scale of what each side was willing to commit, not identical accounting. The narrower claim that survives either way: Chinese private capital writes cheques at the same order of magnitude as the Chinese state, and Indian private capital does not.
