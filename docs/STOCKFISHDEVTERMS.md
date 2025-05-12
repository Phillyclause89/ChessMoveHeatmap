### List of Acronyms

* **General Chess**:

  * [`SF`](https://stockfishchess.org/) – Stockfish, an open‑source UCI‑compatible chess engine ([Stockfish][1])
  * [`LC0`](https://lczero.org/) – Leela Chess Zero, an open‑source neural‑network chess engine ([Wikipedia][2])
  * [`FEN`](https://www.chessprogramming.org/Forsyth-Edwards_Notation) – Forsyth–Edwards Notation for describing chess positions ([ChessProgramming][3])
  * [`PGN`](https://www.chessprogramming.org/Portable_Game_Notation) – Portable Game Notation for recording game moves ([ChessProgramming][4])
  * [`FRC`](https://en.wikipedia.org/wiki/Chess960) – Chess960 (Fischer Random Chess) variant ([Wikipedia][5])
  * [`DFRC`](https://en.wikipedia.org/wiki/Chess960) – Double Fischer Random Chess, a variant of FRC used in certain competitions like TCEC ([Wikipedia][5])
  * [`TC`](https://en.wikipedia.org/wiki/Time_control) – Time control mechanisms for games ([Wikipedia][6]). Can also refer to TalkChess forums in some contexts.
  * [`50MR`](https://www.chessprogramming.org/Fifty-move_Rule) – Fifty‑move rule for draw conditions ([ChessProgramming][7])
  * [`CCC`](https://www.chess.com/computer-chess-championship) – Chess.com Computer Chess Championship ([Chess.com][8]). Sometimes referred to as `CCCC`.
  * [`TCEC`](https://en.wikipedia.org/wiki/Top_Chess_Engine_Championship) – Top Chess Engine Championship ([Wikipedia][9])
  * [`CCRL`](https://www.chessprogramming.org/CCRL) – Computer Chess Rating Lists ([ChessProgramming][10])
  * [`CPW`](https://www.chessprogramming.org/Main_Page) – Chess Programming Wiki, a comprehensive engine‑dev reference ([ChessProgramming][11])
  * [`OB`](https://github.com/AndyGrant/OpenBench) – OpenBench, an open‑source UCI engine‑testing framework ([GitHub][12])

* **General Engines**:

  * [`SSS`](https://www.chessprogramming.org/Small_Sample_Size) – Small sample size in testing ([ChessProgramming][13])
  * [`UCI`](https://www.chessprogramming.org/UCI) – Universal Chess Interface protocol ([ChessProgramming][14])
  * [`SPRT`](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test) – Sequential Probability Ratio Test for engine testing ([ChessProgramming][13])
  * [`LLR`](https://www.chessprogramming.org/Log_Likelihood_Ratio) – Log‑Likelihood Ratio in SPRT tests
  * [`SPSA`](https://www.chessprogramming.org/SPSA) – Simultaneous Perturbation Stochastic Approximation
  * [`NPS`](https://www.chessprogramming.org/Nodes_per_Second) – Nodes per second performance metric
  * [`STC/LTC/VLTC`](https://www.chessprogramming.org/Time_Controls) – Short/Long/Very‑Long time controls
  * [`SB`](https://www.chessprogramming.org/Superbatch) – Superbatch parameter for bulk testing
  * [`EAS`](https://www.chessprogramming.org/Engine_Aggressiveness_Score) – Engine Aggressiveness Score

* **Search**:

  * [`TM`](https://www.chessprogramming.org/Time_Management) – Time management heuristics ([ChessProgramming][15])
  * [`PV`](https://www.chessprogramming.org/Principal_Variation) – Principal variation
  * [`TT`](https://www.chessprogramming.org/Transposition_Table) – Transposition table
  * [`PVS`](https://www.chessprogramming.org/Principal_Variation_Search) – Principal variation search. Not grouped with ZWS due to differing search use cases.
  * [`ID`](https://www.chessprogramming.org/Iterative_Deepening) – Iterative deepening
  * [`IID`](https://www.chessprogramming.org/Internal_Iterative_Reductions) – Internal iterative deepening
  * [`QS`](https://www.chessprogramming.org/Quiescence_Search) – Quiescence search
  * [`SEE`](https://www.chessprogramming.org/Static_Exchange_Evaluation) – Static exchange evaluation. SEE is not classified as a heuristic because it evaluates material gain/loss statically without deeper heuristics.
  * [`ZWS`](https://www.chessprogramming.org/Zero_Window_Search) – Zero‑window search

* **Heuristics**:

  * [`IIR`](https://www.chessprogramming.org/Internal_Iterative_Reductions) – Internal iterative reductions
  * [`RFP`](https://www.chessprogramming.org/Reverse_Futility_Pruning) – Reverse futility pruning. Also referred to as `SNMP` (Static Null Move Pruning).
  * [`FP/FFP`](https://www.chessprogramming.org/Futility_Pruning) – Forward futility pruning
  * [`NMP`](https://www.chessprogramming.org/Null_Move) – Null move pruning
  * [`LMP`](https://www.chessprogramming.org/Late_Move_Pruning) – Late move pruning
  * [`LMR`](https://www.chessprogramming.org/Late_Move_Reduction) – Late move reductions
  * [`ZWS`](https://www.chessprogramming.org/Zero_Window_Search) – Zero‑window search
  * [`GHI`](https://www.chessprogramming.org/Graph_History_Interaction) – Graph history interaction
  * [`SE`](https://www.chessprogramming.org/Singular_Extensions) – Singular extensions
  * [`SMP`](https://www.chessprogramming.org/Symmetric_Multiprocessing) – Symmetric multiprocessing
  * [`MCTS`](https://www.chessprogramming.org/Monte-Carlo_Tree_Search) – Monte Carlo Tree Search ([ChessProgramming][16])
  * [`PUCT`](https://www.chessprogramming.org/PUCT) – Predictor + Upper Confidence bound for Trees

* **Move Ordering**:

  * [`MVV-LVA`](https://www.chessprogramming.org/MVV-LVA) – Most Valuable Victim – Least Valuable Aggressor
  * [`HH`](https://www.chessprogramming.org/History_Heuristic) – History heuristic
  * [`PCM`](https://www.chessprogramming.org/Prior_Countermove_Heuristic) – Prior counter‑move
  * [`CMH`](https://www.chessprogramming.org/Countermove_Heuristic) – Counter move history
  * [`FUH`](https://www.chessprogramming.org/Follow-up_History) – Follow up history
  * [`HP`](https://www.chessprogramming.org/History_Pruning) – History pruning

* **Evaluation**:

  * [`NNUE`](https://www.chessprogramming.org/NNUE) – Efficiently Updatable Neural Network ([ChessProgramming][17])
  * [`HCE`](https://www.chessprogramming.org/Hand-crafted_Evaluation) – Hand‑crafted evaluation
  * [`PST/PSQT`](https://www.chessprogramming.org/Piece-Square_Table) – Piece‑square tables
  * [`BAE`](https://www.chessprogramming.org/Big-Array_Eval) – Big‑array evaluation
  * [`RFB`](https://www.chessprogramming.org/Rook_Forward_Bonus) – Rook forward bonus
  * [`KS`](https://www.chessprogramming.org/King_Safety) – King safety considerations
  * [`STM`](https://www.chessprogramming.org/Side_to_Move) – Side-to-move

---

> *This is **not** exhaustive. See the CPW “Dictionary” or its [Acronym category](https://www.chessprogramming.org/Category:Acronym) for more.*

---
[1]: https://stockfishchess.org/?utm_source=chatgpt.com "Stockfish - Strong open-source chess engine"
[2]: https://en.wikipedia.org/wiki/Leela_Chess_Zero?utm_source=chatgpt.com "Leela Chess Zero"
[3]: https://www.chessprogramming.org/Forsyth-Edwards_Notation?utm_source=chatgpt.com "Forsyth-Edwards Notation - Chessprogramming wiki"
[4]: https://www.chessprogramming.org/index.php?mobileaction=toggle_view_desktop&title=Portable_Game_Notation&utm_source=chatgpt.com "Portable Game Notation - Chessprogramming wiki"
[5]: https://en.wikipedia.org/wiki/Chess960?utm_source=chatgpt.com "Chess960"
[6]: https://en.wikipedia.org/wiki/Time_control?utm_source=chatgpt.com "Time control"
[7]: https://www.chessprogramming.org/Fifty-move_Rule?utm_source=chatgpt.com "Fifty-move Rule - Chessprogramming wiki"
[8]: https://www.chess.com/computer-chess-championship?utm_source=chatgpt.com "Computer Chess Championship"
[9]: https://en.wikipedia.org/wiki/Top_Chess_Engine_Championship?utm_source=chatgpt.com "Top Chess Engine Championship"
[10]: https://www.chessprogramming.org/CCRL?utm_source=chatgpt.com "CCRL - Chessprogramming wiki"
[11]: https://www.chessprogramming.org/Main_Page?utm_source=chatgpt.com "Chess Programming Wiki"
[12]: https://github.com/AndyGrant/OpenBench?utm_source=chatgpt.com "AndyGrant/OpenBench - GitHub"
[13]: https://www.chessprogramming.org/Sequential_Probability_Ratio_Test?utm_source=chatgpt.com "Sequential Probability Ratio Test - Chessprogramming wiki"
[14]: https://www.chessprogramming.org/UCI?utm_source=chatgpt.com "UCI - Chessprogramming wiki"
[15]: https://www.chessprogramming.org/Time_Management?utm_source=chatgpt.com "Time Management - Chessprogramming wiki"
[16]: https://www.chessprogramming.org/Monte-Carlo_Tree_Search?utm_source=chatgpt.com "Monte-Carlo Tree Search - Chessprogramming wiki"
[17]: https://www.chessprogramming.org/NNUE?utm_source=chatgpt.com "NNUE - Chessprogramming wiki"
