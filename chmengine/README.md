# chmengine

The `chmengine` module is an Easter egg component of
the [ChessMoveHeatmap](https://github.com/Phillyclause89/ChessMoveHeatmap) project. This module provides a chess engine
that uses the data produced by the heatmaps to pick moves and allows for playing or training the engines.

## Usage - Play CMHMEngine _(aka: Cmhmey Sr.)_

Below is an example of how to use the `chmengine` module to play a game against the `CMHMEngine` engine using
the `PlayCMHMEngine` class.

```python
from chmengine import PlayCMHMEngine

# Initialize the PlayCMHMEngine
game_manager = PlayCMHMEngine(player_name="Phillyclause89")

# Play a game against the engine, CMHMEngine picks moves via various definite algorithms 
# Use pick_by param to specify what arg gets passed to the CMHMEngine instance.
game_manager.play(pick_by="delta")
```

## Usage - Play CMHMEngine2 _(aka: Cmhmey Jr.)_

Below is an example of how to use the `chmengine` module to play a game against the `CMHMEngine2` engine using
the `PlayCMHMEngine` class.

```python
from chmengine import PlayCMHMEngine, CMHMEngine2

# Initialize the PlayCMHMEngine
game_manager = PlayCMHMEngine(engine=CMHMEngine2, player_name="Phillyclause89", player_color='black')

# Cmhmey Jr. only has one pick_by algorithm that evolves through reinforcement learning updates
game_manager.play(pick_by="CMHMEngine2")
```

## Usage - Train CMHMEngine2 _(aka: Cmhmey Jr.)_

Below is an example of how to use the `chmengine` module to train the `CMHMEngine2` engine using the `PlayCMHMEngine`
class.

```python
from chmengine import PlayCMHMEngine, CMHMEngine2

# Initialize the PlayCMHMEngine
game_manager = PlayCMHMEngine(engine=CMHMEngine2)

# Cmhmey Jr. only has one pick_by algorithm that evolves through reinforcement learning updates
game_manager.train_cmhmey_jr(training_games=1)  # 1 training game is expected to take about ~45 minutes 
```

Click the image below to watch Cmhmey Jr. Train On YouTube!
<p align="center">
  <a href="https://www.youtube.com/live/_-JySFYZhjU?si=fsapzEKLV8CTVYrt" target="_blank">
    <img src="../docs/images/CmhmeyJrTrainingForever.gif" alt="Watch Cmhmey Jr. Train On YouTube!" width="1000">
  </a>
</p>

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

---

Feel free to adjust any sections as needed to better fit the specifics of your project!
