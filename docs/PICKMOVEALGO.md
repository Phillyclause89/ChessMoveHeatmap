# **README: Algorithm Overview for `CMHMEngine2.pick_move`**

The `pick_move` method in `CMHMEngine2` is a sophisticated algorithm that combines **heatmap-based evaluations**, **Q-table integration**, and **conditional recursion** to select the best move in a chess position. This document provides a detailed, academic-style explanation of the algorithm, including its data flow, recursive structure, and decision-making process.

---

## **Algorithm Overview: `CMHMEngine2.pick_move`**

The `pick_move` method evaluates all legal moves from the current board state, assigns scores to each move, and selects the best move based on the evaluation. The scoring process incorporates:

1. **Heatmap-based evaluations**: Static evaluations of board positions using discounted move intensities.
2. **Q-table integration**: Cached evaluations of previously encountered positions.
3. **Conditional recursion**: Deeper analysis of future moves, triggered only for checks or captures.

The algorithm ensures that the selected move is optimal for the current player (White or Black) by considering the opponent's best possible responses.

---

### **1️⃣ Move Evaluation Pipeline**

The `pick_move` method follows a structured pipeline to evaluate and rank moves:

#### **Step 1: Retrieve Legal Moves**

The algorithm begins by retrieving all legal moves for the current board state:

- Let \( M \) represent the set of all legal moves for the current board state \( B \). The algorithm computes \( M \) as follows:

- \[
M = \text{LegalMoves}(B)
\]

- If \( M = \emptyset \), the algorithm raises an exception, indicating that the game has reached a terminal state (e.g., checkmate or stalemate).

- If no legal moves are available, a `ValueError` is raised, indicating the game has ended (e.g., checkmate or stalemate).

#### **Step 2: Evaluate Each Move**

For each legal move, the algorithm:

1. Simulates the move on a copy of the board.
2. Evaluates the resulting board position using the following process:
    - **Recursive Min-Max Lookahead**: The algorithm performs a recursive evaluation of moves at every call to `pick_move`. This recursive process continues until the final stack frame of the recursion path is reached (typically 2 plies deep, or 3 plies in the event of a detected capture or check on ply 2).  
        - **Q-Table Lookup**: At the final stack frame of the recursion, the algorithm checks if a Q-value exists for the resulting position. If a Q-value is available, it is used as the evaluation score.  
        - **Static Heatmap Evaluation**: If no Q-value is available for the final position in the recursion path, the algorithm calculates a score using a static heatmap evaluation via the `calculate_white_minus_black_score` function.

This process is handled by the `_update_current_pick_score_` and `_get_or_calculate_responses_` methods.

#### **Step 3: Rank Moves**

The evaluated moves are currently ranked using an ordered insertion strategy during the evaluation process. However, we plan to transition to a more streamlined approach in the future, where moves will be sorted once at the end of the pipeline.

#### **Step 4: Select the Best Move**

The algorithm selects the best move from the ranked list. If multiple moves have the same highest score, one is chosen randomly to introduce variability:

The algorithm identifies the best move by selecting the highest-ranked move from the ordered list of evaluated moves. For White's turn, the move with the maximum score is chosen, while for Black's turn, the move with the minimum score is selected.

To introduce variability and avoid deterministic behavior, the algorithm incorporates a mechanism to randomly select among equally scored moves. This ensures that the engine does not always follow the same sequence of moves in identical positions, enhancing its unpredictability in gameplay.

The selection process can be summarized as follows:

1. Identify the best move based on the player's turn:
    - For White: Select the move with the highest score.
    - For Black: Select the move with the lowest score.
2. If multiple moves share the same score, randomly select one from the set of equally scored moves.

This approach balances optimal decision-making with an element of randomness, making the engine's playstyle more dynamic and less predictable.

#### **Step 5: Update Q-Table**

The Q-value for the selected move is updated in the Q-table to reflect its evaluation score. The Q-table update process involves two key steps:

1. **Preliminary Q-Value Write**:  
    When a new board position is encountered for the first time, the algorithm calculates a preliminary Q-value for that position. This value is based on the static evaluation or recursive scoring performed during the move evaluation process. The preliminary Q-value is immediately written to the Q-table to ensure that the position is recorded for future reference.

2. **Backpropagation of Scores**:  
    After the best move is selected, the algorithm backpropagates the chosen move's score through the Q-table across all plies in the current recursion layer. This ensures that the Q-values for previously visited positions are updated to reflect the evaluation of the selected move. By propagating the score backward, the algorithm refines its understanding of the position's value, improving the accuracy of future evaluations.

This two-step process allows the Q-table to serve as both a cache for previously evaluated positions and a dynamic learning mechanism that adapts based on the outcomes of recursive evaluations.

---

### **2️⃣ Conditional Recursion in `_get_or_calculate_responses_`**

The `_get_or_calculate_responses_` method evaluates the opponent's possible responses to a candidate move. It works in tandem with `_get_or_calc_response_move_scores_`, which handles the evaluation of individual response moves. Together, these methods implement a **conditionally recursive evaluation** strategy, where recursion is triggered only for specific types of moves (checks or captures).

#### **Recursive Logic**

The recursion in `_get_or_calc_response_move_scores_` is triggered only when:

1. The `go_deeper` flag is set to `True` on the parent call of the potentially recursive flow.  
2. And the move results in a **check** or **capture** on the resulting board state.  

- If the nested check in step 2 is `True`, the algorithm performs one level of recursion with the `go_deeper` flag set to `False`.
- This ensures that the maximum number of recursive calls is 1, while the minimum is 0 if no checks or captures are detected in the response board state.

This flow limits the recursion to tactically significant positions, reducing computational overhead while still capturing critical tactical nuances.

---

### **3️⃣ Scoring System**

The scoring system follows these principles:

- **Positive Scores**: Favorable for White.
- **Negative Scores**: Favorable for Black.
- **Recursive Depth**: The `depth` parameter controls how many future moves are considered during evaluation. Higher depths provide more accurate evaluations but increase computational cost.

#### **Static Evaluation**

If no Q-value is available for a position, the algorithm falls back to a static evaluation using the `calculate_white_minus_black_score` function. This function computes a score based on:

1. **Heatmap Intensity Differences**: The difference in move intensities for White and Black.
2. **King Box Pressure**: The difference in move intensities around the kings.

---

### **4️⃣ Advantages of the Algorithm**

1. **Combines Multiple Evaluation Methods**: Heatmap evaluations, Q-table lookups, and recursive scoring provide a robust and flexible evaluation framework.
2. **Handles Complex Positions**: Recursive scoring allows the engine to analyze deeper tactical sequences.
3. **Efficient Learning**: The Q-table enables the engine to "learn" from previous games and improve over time.

---

### **5️⃣ Limitations**

1. **Computational Cost**: Recursive scoring can be computationally expensive, especially at higher depths.
2. **Randomness**: The random selection of equally scored moves can lead to non-deterministic behavior.
3. **Dependence on Q-Table**: The engine's performance depends on the quality and size of the Q-table.
4. **Limited Tactical Depth**: The conditional recursion strategy may miss deeper tactical sequences beyond the predefined recursion depth.
5. **Static Evaluation Bias**: The fallback static evaluation may not fully capture dynamic positional factors, leading to suboptimal move choices in certain scenarios.

---

### **6️⃣ Summary of Algorithm Flow**

- **Retrieve Legal Moves**: Identify all legal moves for the current board state. If no moves are available, the game is in a terminal state (e.g., checkmate or stalemate).
- **Evaluate Moves**: Simulate each move, evaluate the resulting board position using recursive scoring, Q-table lookups, or static heatmap evaluations.
- **Rank Moves**: Rank the evaluated moves based on their scores, using an ordered insertion strategy.
- **Select Best Move**: Choose the highest-scoring move for White or the lowest-scoring move for Black. Introduce randomness if multiple moves share the same score.
- **Update Q-Table**: Record the evaluation score of the selected move in the Q-table and backpropagate the score through the recursion layers.

This flow ensures a balance between computational efficiency and strategic depth, enabling the engine to make informed and dynamic move choices.

---

### **7️⃣ Final Considerations**

After a game played by the engine is completed, a specialized method is invoked to refine the Q-table and improve future evaluations. This method iterates through the moves of the game in reverse order, leveraging the `pick_move` method to backpopulate scores recursively:

1. **Reverse Move Processing**:  
    The method `pop`s moves out of the board object, effectively rewinding the game to earlier positions. This allows the engine to revisit each position from the game in reverse order.

2. **Reevaluation and Backpropagation**:  
    For each position, the `pick_move` method is recalled to reevaluate the move choices. This ensures that the resulting new choice from the backpopulation process is also backpopulated further up the recursion chain.

3. **Enhanced Learning**:  
    By processing the game in reverse, the engine refines its Q-table entries for all positions encountered during the game. This iterative backpropagation improves the accuracy of the Q-values, enabling the engine to make better decisions in future games.

This post-game analysis step is a critical component of the engine's learning mechanism, ensuring that the Q-table evolves dynamically based on actual gameplay outcomes.

---

This README provides a comprehensive explanation of the `pick_move` algorithm in `CMHMEngine2`, highlighting its recursive structure, scoring system, and integration with Q-tables. Let me know if you'd like further refinements!
