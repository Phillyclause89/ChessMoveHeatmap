Here is an updated README draft that focuses on the `PlayCMHMEngine` class, which is the main entry point into the `chmengine` module:

---

# chmengine

## Usage - Play CMHMEngine _(aka: Cmhmey Sr.)_

Below is an example of how to use the `chmengine` module to play a game against the `CMHMEngine` engine using the `PlayCMHMEngine` class.

```python
from chmengine import PlayCMHMEngine

# Initialize the PlayCMHMEngine
game_manager = PlayCMHMEngine(player_name="Phillyclause89")

# Play a game against the engine, CMHMEngine picks moves via varius definite algorithms 
# Use pick_by param to specify what arg gets passed to the CMHMEngine instance.
game_manager.play(pick_by="delta")
```

## Usage - Play CMHMEngine2 _(aka: Cmhmey Jr.)_

```python
from chmengine import PlayCMHMEngine, CMHMEngine2

# Initialize the PlayCMHMEngine
game_manager = PlayCMHMEngine(engine=CMHMEngine2, player_name="Phillyclause89", player_color='black')

# Cmhmey Jr. only has one pick_by algorithm that evolves through reinforcement learning updates
game_manager.play(pick_by="CMHMEngine2")
```

## Usage - Train CMHMEngine2 _(aka: Cmhmey Jr.)_

```python
from chmengine import PlayCMHMEngine, CMHMEngine2

# Initialize the PlayCMHMEngine
game_manager = PlayCMHMEngine(engine=CMHMEngine2)

# Cmhmey Jr. only has one pick_by algorithm that evolves through reinforcement learning updates
game_manager.train_cmhmey_jr(training_games=1)  # 1 training game is expected to take about ~45 minutes 
```

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.