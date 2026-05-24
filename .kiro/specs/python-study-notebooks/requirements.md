# Requirements Document

## Introduction

This feature creates a structured 100-day Python learning curriculum delivered as Jupyter Notebook (.ipynb) files. The curriculum is organized into 100 folders on drive D, each containing one standalone, self-contained notebook covering a specific Python topic. The notebooks progress from absolute beginner fundamentals through intermediate and advanced topics, culminating in applied data science, machine learning, and software engineering skills. Each notebook is designed to be studied in approximately one day and requires no external dependencies beyond what is explicitly introduced in that notebook.

## Glossary

- **Curriculum**: The full set of 100 notebooks forming the complete 100-day Python learning path.
- **Notebook**: A single Jupyter `.ipynb` file containing markdown explanations, code cells, and exercises for one topic.
- **Day_Folder**: A directory named in the format `NN_py` (e.g., `01_py`, `02_py`, ..., `100_py`) that contains exactly one Notebook.
- **Root_Directory**: The directory `D:\python study notebooks` that contains all 100 Day_Folders.
- **Topic**: A specific Python concept or library covered by a single Notebook.
- **Phase**: A logical grouping of consecutive days sharing a common learning theme (e.g., Beginner, Intermediate, Advanced).
- **Standalone**: A Notebook that can be opened and executed independently without requiring outputs or state from other Notebooks.
- **Exercise**: A code cell within a Notebook that prompts the learner to write or complete code to reinforce the Topic.
- **Solution**: A code cell or markdown section within a Notebook that provides the correct answer to an Exercise.
- **Kernel**: The Python execution environment used by Jupyter to run Notebook cells.

---

## Requirements

### Requirement 1: Root Directory and Folder Structure

**User Story:** As a learner, I want a clearly organized folder structure on drive D, so that I can easily navigate to any day's material.

#### Acceptance Criteria

1. THE Curriculum SHALL be stored under the Root_Directory `D:\python study notebooks`.
2. THE Root_Directory SHALL contain exactly 100 Day_Folders.
3. THE Day_Folder names SHALL follow the pattern `NN_py` where `NN` is a zero-padded two-digit number for days 01–99 and a three-character string `100` for day 100 (i.e., `01_py`, `02_py`, ..., `09_py`, `10_py`, ..., `99_py`, `100_py`).
4. WHEN the Day_Folders are sorted lexicographically, THE Root_Directory SHALL present them in ascending day order from `01_py` to `100_py`.
5. EACH Day_Folder SHALL contain exactly one `.ipynb` file.

---

### Requirement 2: Notebook File Naming

**User Story:** As a learner, I want each notebook file to have a descriptive name, so that I can identify the topic at a glance without opening the file.

#### Acceptance Criteria

1. THE Notebook file name SHALL reflect the Topic covered on that day (e.g., `variables_and_types.ipynb`, `list_comprehensions.ipynb`).
2. THE Notebook file name SHALL use lowercase letters and underscores only (snake_case), with no spaces or special characters.
3. THE Notebook file name SHALL end with the `.ipynb` extension.

---

### Requirement 3: Notebook Structure and Content Standards

**User Story:** As a learner, I want each notebook to follow a consistent structure, so that I know what to expect and can study efficiently every day.

#### Acceptance Criteria

1. EACH Notebook SHALL begin with a markdown cell containing the day number, topic title, and a brief (2–4 sentence) description of what will be learned.
2. EACH Notebook SHALL contain at least one markdown cell explaining the core concept(s) of the Topic before any code is introduced.
3. EACH Notebook SHALL contain at least three executable code cells demonstrating the Topic with working Python code and inline comments.
4. EACH Notebook SHALL contain at least two Exercise cells that prompt the learner to write or complete code.
5. EACH Notebook SHALL contain Solution cells or sections corresponding to each Exercise.
6. EACH Notebook SHALL end with a markdown cell summarizing the key takeaways and listing 2–5 suggested next steps or further reading topics.
7. THE Notebook SHALL be Standalone: all imports, variable definitions, and data required to run the Notebook SHALL be defined within the Notebook itself.
8. WHEN a Notebook introduces a new library, THE Notebook SHALL include a markdown cell explaining how to install the library (e.g., `pip install <library>`).

---

### Requirement 4: Curriculum Progression and Topic Coverage

**User Story:** As a learner, I want the 100 notebooks to form a coherent learning path from beginner to advanced, so that each day builds on the previous day's knowledge.

#### Acceptance Criteria

1. THE Curriculum SHALL be divided into four Phases: Beginner (Days 01–25), Intermediate (Days 26–50), Advanced (Days 51–75), and Applied (Days 76–100).
2. THE Beginner Phase SHALL cover foundational topics including: Python setup and environment, variables and data types, strings and string methods, numbers and arithmetic, lists, tuples, sets, dictionaries, conditionals, loops, functions, scope, modules, file I/O, and exception handling.
3. THE Intermediate Phase SHALL cover topics including: object-oriented programming (classes, inheritance, polymorphism), iterators and generators, decorators, context managers, comprehensions, lambda and functional tools, regular expressions, working with JSON and CSV, virtual environments, and standard library modules (e.g., `os`, `sys`, `datetime`, `collections`, `itertools`).
4. THE Advanced Phase SHALL cover topics including: type hints and mypy, dataclasses, abstract base classes, metaclasses, concurrency (threading, multiprocessing, asyncio), testing with pytest, logging, packaging and distribution, design patterns, and performance profiling.
5. THE Applied Phase SHALL cover topics including: NumPy, Pandas (data loading, cleaning, transformation, aggregation), Matplotlib, Seaborn, Plotly, web scraping with BeautifulSoup/requests, REST API consumption, SQL with SQLite, scikit-learn (supervised and unsupervised learning), and a final capstone project notebook.
6. WHEN a Topic in a later day depends on a concept introduced in an earlier day, THE later Notebook SHALL reference the earlier day number in a markdown cell.
7. THE Curriculum SHALL NOT repeat the same primary Topic across two different Notebooks.

---

### Requirement 5: Code Quality and Correctness

**User Story:** As a learner, I want all code in the notebooks to be correct and runnable, so that I can execute cells without encountering errors.

#### Acceptance Criteria

1. ALL code cells in EACH Notebook SHALL be syntactically valid Python 3 code.
2. WHEN a Notebook is executed from top to bottom with a fresh Kernel, THE Notebook SHALL complete without raising unhandled exceptions.
3. ALL code cells SHALL include inline comments explaining non-obvious logic.
4. THE Notebook SHALL use Python 3.8+ compatible syntax throughout.
5. IF a code cell is intentionally expected to raise an exception (e.g., to demonstrate error handling), THEN THE Notebook SHALL wrap the code in a try/except block or include a markdown cell explicitly stating that an error is expected.

---

### Requirement 6: Beginner Accessibility

**User Story:** As a complete beginner, I want the early notebooks to assume no prior programming knowledge, so that I can start learning Python from scratch.

#### Acceptance Criteria

1. THE Notebooks in the Beginner Phase (Days 01–25) SHALL assume no prior Python or programming knowledge.
2. EACH Notebook in the Beginner Phase SHALL define every new term in a markdown cell before using it in code.
3. THE Day 01 Notebook SHALL include instructions for installing Python and Jupyter Notebook/JupyterLab.
4. THE Day 01 Notebook SHALL explain how to run a Jupyter Notebook cell.

---

### Requirement 7: Advanced and Applied Depth

**User Story:** As an intermediate learner, I want the later notebooks to cover advanced topics in sufficient depth, so that I can develop professional-level Python skills.

#### Acceptance Criteria

1. THE Notebooks in the Advanced Phase (Days 51–75) SHALL include at least one real-world use case or practical example per Topic.
2. THE Notebooks in the Applied Phase (Days 76–100) SHALL include at least one dataset or realistic data scenario per Topic.
3. THE Day 100 Notebook SHALL be a capstone project that integrates concepts from at least three different Phases into a single end-to-end project.
4. WHERE a Topic has commonly known performance considerations, THE Notebook SHALL include a markdown cell or code cell demonstrating the performance implication.

---

### Requirement 8: Notebook Metadata

**User Story:** As a learner using Jupyter, I want each notebook to have correct metadata, so that it opens properly and runs with the correct kernel.

#### Acceptance Criteria

1. EACH Notebook SHALL include valid Jupyter notebook JSON metadata with `nbformat` set to 4 and `nbformat_minor` set to at least 4.
2. EACH Notebook's kernel metadata SHALL specify Python 3 as the kernel language.
3. EACH Notebook SHALL include a `language_info` metadata field specifying `"name": "python"`.

---

### Requirement 9: Curriculum Topic List

**User Story:** As a learner, I want a defined topic for each of the 100 days, so that I can plan my study schedule in advance.

#### Acceptance Criteria

1. THE Curriculum SHALL assign exactly one Topic to each of the 100 days according to the following schedule:

**Beginner Phase (Days 01–25)**
| Day | Topic |
|-----|-------|
| 01 | Python Setup, Environment & First Program |
| 02 | Variables, Data Types & Type Conversion |
| 03 | Strings & String Methods |
| 04 | Numbers, Arithmetic & Math Module |
| 05 | Boolean Logic & Comparison Operators |
| 06 | Lists – Creation, Indexing & Slicing |
| 07 | Lists – Methods & Mutation |
| 08 | Tuples & Immutability |
| 09 | Sets & Set Operations |
| 10 | Dictionaries – Basics |
| 11 | Dictionaries – Advanced Methods |
| 12 | Conditional Statements (if/elif/else) |
| 13 | for Loops & range() |
| 14 | while Loops & Loop Control |
| 15 | List Comprehensions |
| 16 | Functions – Definition & Arguments |
| 17 | Functions – Return Values & Scope |
| 18 | Lambda Functions & map/filter/reduce |
| 19 | Modules & Importing |
| 20 | File I/O – Reading & Writing Text Files |
| 21 | File I/O – Working with CSV Files |
| 22 | Exception Handling – try/except/finally |
| 23 | Exception Handling – Custom Exceptions |
| 24 | Recursion |
| 25 | Beginner Capstone – Mini Project |

**Intermediate Phase (Days 26–50)**
| Day | Topic |
|-----|-------|
| 26 | OOP – Classes & Objects |
| 27 | OOP – Inheritance & Polymorphism |
| 28 | OOP – Encapsulation & Properties |
| 29 | OOP – Dunder/Magic Methods |
| 30 | Iterators & Iterables |
| 31 | Generators & yield |
| 32 | Decorators |
| 33 | Context Managers & with Statement |
| 34 | Dictionary & Set Comprehensions |
| 35 | Regular Expressions (re module) |
| 36 | Working with JSON |
| 37 | Working with CSV (csv module) |
| 38 | datetime & time Modules |
| 39 | os & sys Modules |
| 40 | collections Module |
| 41 | itertools Module |
| 42 | functools Module |
| 43 | Virtual Environments & pip |
| 44 | Sorting & Searching Algorithms |
| 45 | Recursion – Advanced Patterns |
| 46 | String Formatting & f-strings |
| 47 | Pathlib & File System Operations |
| 48 | Logging Module |
| 49 | argparse – Command-Line Arguments |
| 50 | Intermediate Capstone – Mini Project |

**Advanced Phase (Days 51–75)**
| Day | Topic |
|-----|-------|
| 51 | Type Hints & Annotations |
| 52 | Dataclasses |
| 53 | Abstract Base Classes (ABC) |
| 54 | Metaclasses |
| 55 | Slots & Memory Optimization |
| 56 | Threading |
| 57 | Multiprocessing |
| 58 | asyncio – Async/Await Basics |
| 59 | asyncio – Advanced Patterns |
| 60 | Testing with unittest |
| 61 | Testing with pytest |
| 62 | Mocking & Test Fixtures |
| 63 | Packaging & setup.py / pyproject.toml |
| 64 | Design Patterns – Creational |
| 65 | Design Patterns – Structural |
| 66 | Design Patterns – Behavioral |
| 67 | Performance Profiling & cProfile |
| 68 | Memory Profiling & Optimization |
| 69 | Cython & ctypes Basics |
| 70 | Descriptors & Attribute Access |
| 71 | Closures & Scoping Deep Dive |
| 72 | Functional Programming Patterns |
| 73 | Comprehensions – Advanced Patterns |
| 74 | Python Internals – Bytecode & dis |
| 75 | Advanced Capstone – Mini Project |

**Applied Phase (Days 76–100)**
| Day | Topic |
|-----|-------|
| 76 | NumPy – Arrays & Operations |
| 77 | NumPy – Indexing, Slicing & Broadcasting |
| 78 | NumPy – Linear Algebra & Statistics |
| 79 | Pandas – Series & DataFrames |
| 80 | Pandas – Loading & Inspecting Data |
| 81 | Pandas – Data Cleaning |
| 82 | Pandas – Filtering & Querying |
| 83 | Pandas – GroupBy & Aggregation |
| 84 | Pandas – Merging & Joining |
| 85 | Pandas – Time Series |
| 86 | Matplotlib – Basic Plots |
| 87 | Matplotlib – Customization & Subplots |
| 88 | Seaborn – Statistical Visualization |
| 89 | Plotly – Interactive Visualization |
| 90 | Web Scraping with requests & BeautifulSoup |
| 91 | REST API Consumption |
| 92 | SQL with SQLite3 |
| 93 | SQLAlchemy ORM Basics |
| 94 | scikit-learn – Data Preprocessing |
| 95 | scikit-learn – Supervised Learning |
| 96 | scikit-learn – Unsupervised Learning |
| 97 | scikit-learn – Model Evaluation |
| 98 | Natural Language Processing with NLTK/spaCy |
| 99 | Working with Images (Pillow) |
| 100 | Capstone Project – End-to-End Data Pipeline |

