# NeuroShift

The main Repository of the NeuroShift Dashboard.

![Adversarial Input Page](https://i.postimg.cc/x1SBJMQx/image.png)

## The Contributors to this Project are:

* [Ayasfoafs](https://github.com/Ayasfoafs)
* [FlorianREGAZ](https://github.com/FlorianREGAZ)
* [Julius-W](https://github.com/Julius-W)
* [lyessli](https://github.com/lyessli)
* [PiIsRational](https://github.com/PiIsRational)

## Introduction
NNs are powerful tools for solving complex problems by allowing
computers to learn from data. NNs mimic the human brain in the
sense that they process information through interconnected Nodes (neurons) and
can adapt and learn from experiences. This similarity enables them to
recognize patterns, make predictions, and perform tasks like image or
speech recognition, providing a flexible and effective approach to solving a
wide range of problems with Artificial Intelligence.

NNs are increasingly used for very safety critical tasks,
from autonomous driving to cancer detection.
Because safety critical tasks are human centric,
we can't afford for the NN to make mistakes,
like confusing a stop sign or a red traffic light for a yield sign, as it can lead to loss of lifes.
That's why it's crucial to verify and test NNs against different kinds of Perturbations and
analyze how they behave. We want to make sure they understand things correctly and
make the right decisions,
especially when it comes to important things such as driving safely or detecting illnesses.

Our project **NeuroShift**, aims to let users
analyze NNs impacted by a multitude of Perturbations.
We will develop a comprehensive and user-friendly dashboard that serves
as an analytical tool. This dashboard enables users to observe, understand,
and quantify how NNs respond to DDSs, MDSs, ODDs
and Adversarial Attacks.

## Installation

Start by cloning the repository into the wanted `~/<path>` and installing the requirements.
```console
pi@rational:~/<path>$ git clone https://github.com/NeuroShift/NeuroShift
pi@rational:~/<path>$ pip install -r requirements.txt
```

## Running

Running the app will launch the server and the streamlit website should be loaded on the localhost.

To run the site simply execute:
```console
pi@rational:~/<path>$ ./scripts/run
```

## Development Environment

### Preparing Development

To install the base development environment, you need to install the test requirements:
```console
pi@rational:~/<path>$ pip install -r test-requirements.txt
```

### Running The Developement Evironnement

The tests can be run with (`pytest`):
```console
pi@rational:~/<path>$ ./scripts/test
```

To get the code coverage run:
```console
pi@rational:~/<path>$ ./scripts/cover
```

To format the code automatically before linting run:
```console
pi@rational:~/<path>$ ./scripts/format
```

a folder with the name `htmlcov` should have been generated,
it will contain and `index.html` that can be opened with any browser.

The linters (`pylama`):
```console
pi@rational:~/<path>$ ./scripts/lint
```
