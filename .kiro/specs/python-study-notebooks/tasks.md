# Tasks

## Task List

- [x] 1. Set up project structure and generator scaffold
  - [x] 1.1 Create the `generate_notebooks.py` file with entry point, imports, and `main()` stub
  - [x] 1.2 Implement `markdown_cell(source)` and `code_cell(source)` factory functions
  - [x] 1.3 Implement `write_notebook(root, day, filename, notebook)` file writer using `pathlib`
  - [x] 1.4 Implement `get_folder_name(day)` helper that returns `NN_py` / `100_py` correctly

- [x] 2. Implement NotebookBuilder
  - [x] 2.1 Implement `NotebookBuilder.build(entry)` that assembles a complete nbformat 4 notebook dict
  - [x] 2.2 Ensure the first cell is a markdown title cell with day number, phase, topic, and 2–4 sentence description
  - [x] 2.3 Add library install markdown cell when `entry["libraries"]` is non-empty
  - [x] 2.4 Add concept explanation markdown cells followed by demo code cells for each concept
  - [x] 2.5 Add at least 2 exercise code cells with `# Exercise N:` prompt comments
  - [x] 2.6 Add solution markdown header and solution code cells for each exercise
  - [x] 2.7 Add final summary markdown cell with key takeaways and next steps
  - [x] 2.8 Add cross-reference markdown note when `entry["references"]` is non-empty

- [ ] 3. Build the CURRICULUM data structure — Beginner Phase (Days 01–25)
  - [-] 3.1 Define entries for Days 01–10 with concepts, code_examples, exercises, solutions, takeaways
  - [ ] 3.2 Define entries for Days 11–25 with concepts, code_examples, exercises, solutions, takeaways
  - [ ] 3.3 Ensure Day 01 entry includes Python/Jupyter installation instructions and "how to run a cell" content
  - [ ] 3.4 Ensure Day 25 (Beginner Capstone) integrates concepts from Days 01–24

- [ ] 4. Build the CURRICULUM data structure — Intermediate Phase (Days 26–50)
  - [ ] 4.1 Define entries for Days 26–37 with concepts, code_examples, exercises, solutions, takeaways
  - [ ] 4.2 Define entries for Days 38–50 with concepts, code_examples, exercises, solutions, takeaways
  - [ ] 4.3 Ensure Day 50 (Intermediate Capstone) integrates concepts from Days 26–49

- [ ] 5. Build the CURRICULUM data structure — Advanced Phase (Days 51–75)
  - [ ] 5.1 Define entries for Days 51–62 with concepts, code_examples, exercises, solutions, takeaways
  - [ ] 5.2 Define entries for Days 63–75 with concepts, code_examples, exercises, solutions, takeaways
  - [ ] 5.3 Ensure each Advanced entry includes at least one real-world use case in its code_examples
  - [ ] 5.4 Ensure Day 75 (Advanced Capstone) integrates concepts from Days 51–74

- [ ] 6. Build the CURRICULUM data structure — Applied Phase (Days 76–100)
  - [ ] 6.1 Define entries for Days 76–88 (NumPy, Pandas, Matplotlib, Seaborn) with dataset scenarios
  - [ ] 6.2 Define entries for Days 89–99 (Plotly, web scraping, APIs, SQL, scikit-learn, NLP, Pillow)
  - [ ] 6.3 Define Day 100 capstone entry that references concepts from Beginner, Intermediate, Advanced, and Applied phases
  - [ ] 6.4 Ensure all Applied entries include `libraries` field with required pip packages

- [ ] 7. Implement main generation pipeline
  - [ ] 7.1 Add startup validation: detect duplicate filename stems and duplicate topic titles, raise `ValueError` if found
  - [ ] 7.2 Add startup validation: assert curriculum has exactly 100 entries with day numbers 1–100
  - [ ] 7.3 Implement `main()` loop: iterate CURRICULUM, build each notebook, write to disk, print progress
  - [ ] 7.4 Make root directory configurable via command-line argument (default `D:\python study notebooks`)

- [ ] 8. Write property-based tests
  - [ ] 8.1 Set up `hypothesis` test file `tests/test_notebook_properties.py`
  - [ ] 8.2 Write `test_folder_naming` — Property 1: folder count and naming
  - [ ] 8.3 Write `test_one_notebook_per_folder` — Property 2: one notebook per folder
  - [ ] 8.4 Write `test_filename_snake_case` — Property 3: notebook filename is snake_case
  - [ ] 8.5 Write `test_valid_nbformat4_json` — Property 4: valid nbformat 4 JSON round-trip
  - [ ] 8.6 Write `test_cell_structure_invariants` — Property 5: notebook cell structure invariants
  - [ ] 8.7 Write `test_no_cross_notebook_imports` — Property 6: standalone self-containment
  - [ ] 8.8 Write `test_phase_assignment` — Property 7: phase assignment correctness
  - [ ] 8.9 Write `test_no_duplicate_topics` — Property 8: no duplicate primary topics
  - [ ] 8.10 Write `test_code_syntax_validity` — Property 9: code cell syntax validity
  - [ ] 8.11 Write `test_code_cells_have_comments` — Property 10: code cells contain inline comments
  - [ ] 8.12 Write `test_library_install_cell` — Property 11: library install cell present when needed
  - [ ] 8.13 Write `test_cross_reference_links` — Property 12: cross-reference links present when needed
  - [ ] 8.14 Write `test_curriculum_completeness` — Property 13: curriculum completeness

- [ ] 9. Write unit / example-based tests
  - [ ] 9.1 Write test verifying Day 01 notebook contains installation instructions and "how to run a cell" explanation
  - [ ] 9.2 Write test verifying Day 100 notebook references concepts from at least 3 different phases
  - [ ] 9.3 Write test verifying a notebook with libraries (e.g., Day 76) contains a `pip install` markdown cell
  - [ ] 9.4 Write test verifying Day 25, 50, 75 capstone notebooks reference earlier day numbers

- [ ] 10. Integration test and final validation
  - [ ] 10.1 Write integration test that runs `main()` against a temp directory and verifies exactly 100 folders are created
  - [ ] 10.2 Verify all 100 generated `.ipynb` files parse as valid JSON with `nbformat == 4`
  - [ ] 10.3 Run all property-based and unit tests and confirm they pass
  - [ ] 10.4 Run `generate_notebooks.py` to produce the final output under `D:\python study notebooks`
