

Dear Jordan,

Thank you for submitting your paper, Saturation and Influence Dynamics in Progressive Second Price Markets, to ICDCS 2025. We appreciate your contribution and the effort you have put into your research.

After a thorough review process, we regret to inform you that we are unable to accept your paper for presentation/publication at this time. This year we received a total of 529 valid submissions, and among these, only 104 papers are accepted. The decision was based on feedback from our reviewers, who carefully evaluated all submissions. Due to the highly competitive nature of the selection process, we had to make difficult choices among many strong submissions.

We encourage you to review the feedback provided by the reviewers, as it may be helpful for refining your work. We sincerely appreciate your interest in ICDCS 2025 and hope you will consider submitting your research to the conference in the future.

Thank you again for your contribution, and we wish you the best in your future research endeavors.

Best regards,

TPC Co-Chairs
Songqing Chen
Neeraj Suri

SUBMISSION: 469
TITLE: Saturation and Influence Dynamics in Progressive Second Price Markets


----------------------- REVIEW 1 ---------------------
SUBMISSION: 469
TITLE: Saturation and Influence Dynamics in Progressive Second Price Markets
AUTHORS: Jordan Blocher and Frederick Harris

----------- Reviewer's expertise -----------
SCORE: 3 (familar)
----------- Paper strengths -----------
This paper studies Progressive Second Price auctions in a decentralized market, represented by a graph, where each node/agent has a deterministically limited area of influence. The authors introduce graph-theoretical notions — such as influence sets and saturated components — into the auction graph and study the asymptotic behavior of the auction dynamic.
----------- Paper weaknesses -----------
The topic of the paper is potentially interesting, but the lack of a clear statement of the model and the problem makes it difficult to review.
----------- Detailed review -----------
This paper studies Progressive Second Price auctions in a decentralized market, represented by a graph, where each node/agent has a deterministically limited area of influence. The authors introduce graph-theoretical notions — such as influence sets and saturated components — into the auction graph and study the asymptotic behavior of the auction dynamic.

The topic of the paper is potentially interesting, but the lack of a clear statement of the model and the problem makes it difficult to review. How does the auction take place? The nodes correspond to agents, but how do they interact with each other? What is the allocation and payment rule? Moreover, what is the dynamic? What is V’ in the definition of a saturated component? Besides the ambiguity in the model, the technical part mainly entails verifying (or assuming) that the hypothesis of Banach Fixed-Point Theorem are respected.
----------- Your overall rating -----------
SCORE: -2 (reject)



----------------------- REVIEW 2 ---------------------
SUBMISSION: 469
TITLE: Saturation and Influence Dynamics in Progressive Second Price Markets
AUTHORS: Jordan Blocher and Frederick Harris

----------- Reviewer's expertise -----------
SCORE: 1 (unfamiliar)
----------- Paper strengths -----------
?
----------- Paper weaknesses -----------
Not readable for a non-expert.
----------- Detailed review -----------
I am very sorry but this paper is not readable for a non-expert. It is written as if Progressive Second Price Auctions as well as numerous concepts related to them are well-known to the reader. The paper does not introduce the reader to the general setting, it does not define key concepts and notions that it uses, and it does not even explain the contributions in the introductory parts of the paper.

I started following references, but they also did not make much sense. For example, I got lost in the definition of saturated components on p. 1, and noticed there the sentence "Saturated components play a crucial role in defining the stability of the strategy space and ensuring that auction dynamics converge over time [7]." I started to read ref. [7], and it did not say a word about saturated components, stability, strategies, auctions, dynamics, or convergence. I really had to give up.

Lack of space is not an excuse, as this is a 6-page submission and the page limit is 11 pages. There might be something interesting and exciting in this work, but it is impossible for me to say what. Some major revision is needed to make this suitable for the ICDCS audience.
----------- Your overall rating -----------
SCORE: -2 (reject)



----------------------- REVIEW 3 ---------------------
SUBMISSION: 469
TITLE: Saturation and Influence Dynamics in Progressive Second Price Markets
AUTHORS: Jordan Blocher and Frederick Harris

----------- Reviewer's expertise -----------
SCORE: 3 (familar)
----------- Paper strengths -----------
1. The paper offers a robust theoretical framework for the Progressive Second Price (PSP) auction mechanism.
2. The paper provides formal mathematical proofs for stability and convergence in PSP auctions.
3. The paper highlights the adaptability of PSP auctions to varying supply-demand ratios and bounded rationality.
4. It contrasts PSP with traditional centralized methods like Vickrey auctions, showing its practical advantages.
----------- Paper weaknesses -----------
1. The real-world implementation challenges of PSP auctions are not deeply explored.
2. The paper could benefit from more discussion on how PSP performs in various market conditions.
3. The paper assumes bounded rationality but does not explore different models of irrational behavior.
----------- Detailed review -----------
The Progressive Second Price (PSP) auction is a decentralized auction mechanism designed to allocate resources efficiently while maintaining properties like truthfulness, efficiency, and optimality. PSP extends traditional auction theory (e.g., Vickrey auctions) into dynamic, decentralized markets. Participants (buyers and sellers) adjust their bids progressively over time, interacting through a network structure rather than a centralized platform. While PSP auctions have desirable theoretical properties, several challenges arise when implementing them in real-world, decentralized systems. Common problems that arise in PSPs and are studied in this paper are the following:
1. Influence Propagation: Understanding how participants' decisions influence others through a network graph.
2. Saturation: Identifying when interactions within the auction graph stabilize.
3. Convergence and Robustness: Ensuring the auction mechanism reaches equilibrium under realistic conditions, including bounded rationality and dynamic demand.


The authors leverage graph theory to define and bound influence dynamics in PSP auctions, proposing the following concepts and tools:
1. Influence Sets:
   - Primary Influence Set (\lambda): Represents direct neighbors a node (buyer/seller) influences in a single auction iteration.
   - Secondary Influence Set (\gamma): Captures indirect ripple effects through the network over multiple iterations.
2. Reachability Bounds: A deterministic limit \(B\) that constrains influence propagation within a finite graph region, ensuring stability and preventing uncontrolled spread.
3. Saturation Components: Graph substructures where additional bids or edges no longer impact outcomes, marking equilibrium in the system.
4. Convergence Proof: Demonstrates that influence sets \lambda and \gamma stabilize under the reachability constraint, ensuring robust and predictable outcomes.
5. Graph Morphisms: Models transitions between saturated states under varying resource conditions (e.g., abundance vs. scarcity), adapting to dynamic supply-demand ratios (SDR).

In this paper, the authors prove the stability and convergence of the PSP auction framework using graph-theoretic principles and formal mathematical analysis. The authors demonstrate that influence propagation, modeled through primary and secondary influence sets, is bounded by a deterministic reachability constraint, ensuring that the system operates within a finite and predictable strategy space. They introduce the concept of saturation, where the auction reaches equilibrium, and formally prove that influence sets converge to stable states as auction iterations progress. The convergence proofs rely on contraction mapping principles and the bounded nature of influence propagation, guaranteeing that noise and external perturbations do not destabilize the auction dynamics. These results establish that the PSP mechanism is robust, with equilibrium states remaining consistent over time. Compared to traditional methods like Vickrey or VCG auctions, which assume centralized settin
gs and perfect rationality, and heuristic-based decentralized approaches, the PSP framework stands out for its theoretical guarantees, adaptability to varying supply-demand ratios, and incorporation of realistic bounded rationality behaviors. This makes it a more reliable and practical solution for decentralized resource allocation in dynamic environments.

Overall, this paper makes a compelling theoretical contribution to the field of decentralized resource allocation. By rigorously formalizing the stability and convergence properties of PSP auctions through graph-theoretic constructs, the authors advance the state-of-the-art beyond traditional centralized mechanisms and ad-hoc decentralized approaches. The proofs of bounded influence propagation and saturation provide actionable insights for designing practical distributed systems where equilibrium and robustness are critical (e.g., edge computing, IoT markets, or federated learning ecosystems). While experimental validation of the framework in real-world scenarios remains future work, the theoretical foundations laid here are both novel and impactful. The paper’s clarity, mathematical depth, and relevance to dynamic distributed systems align with ICDCS’s scope. Therefore, I recommend acceptance
----------- Your overall rating -----------
SCORE: 2 (accept)

Next Steps for Immediate Revision:

1. Rewrite the introduction and abstract clearly stating the problem, significance, and contributions.


2. Create a detailed "Auction Model" section.


3. Explicitly state and apply the Banach Fixed-Point Theorem in the "Convergence" section.


4. Review and correct citations.


5. Include brief discussions of practical considerations and alternative models of bounded rationality.



Key Priorities Based on Reviewer Feedback:

1. Clarity of Model and Auction Rules:

Reviewer 1 and 2 specifically noted the lack of a clear, standalone model description and difficulty understanding the mechanism's rules.

We'll move and clarify the model and mechanism (now spread between Sections 3 and 4) into a dedicated “Auction Mechanism” section earlier in the paper.

We’ll include definitions of players, bid structure, timing, pricing rules, and interactions all in one place.



2. Define Key Concepts Early:

Reviewer 2 was lost at the mention of “saturated components” on page 1. Let’s introduce influence sets, saturation, and graph morphisms clearly in the abstract and the intro, not just in later sections.

We might also consider a figure early in the paper to visually show PSP dynamics, influence propagation, and graph structure.



3. Improve Readability for Non-experts:

Add a simplified example or illustration (perhaps in the influence section) showing a toy PSP auction graph.

Use more intuitive naming or provide footnotes/glossary-style side comments for terms like $\pi$, $\omega$, $\Lambda$, etc.



4. Reorganize for Coherence:

Consider breaking the paper into the following structure:

1. Introduction and Motivation (clearly state problem and contributions)


2. PSP Auction Mechanism (rules, structure, pricing)


3. Graph Model and Influence Sets (primary, secondary, propagation)


4. Saturation and Convergence


5. Equilibrium and Sensitivity


6. Conclusion



