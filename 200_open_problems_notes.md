# Problems (candidates for MJ MATS)

## The Case for Analysing Toy Language Models

### Understanding neurons:

- **1.1 Deeply reverse engineer a neuron in a 1 layer (1L) model.**

- **1.2 Find an interesting neuron in the model that maybe represents some feature.**
  - Can I fully reverse engineer which direction in the model should activate that feature (ie, as calculated from the embedding and attention, in the residual stream in the middle of the layer) and compare it to the neuron input direction?

## Looking for Circuits in the Wild

### Circuits in natural language

- **B\*
2.2 Continuing common sequences**
  - not sequences that have already occurred in the prompt, but sequences that are common in natural language.
  - E.g. `“1 2 3 4” -> “ 5”`, `“1, 2, 3, 4,”->” 5”`, `“Monday\nTuesday\n”->”Wednesday”`, `“I, II, III, IV,” -> “V”`, etc

- **B\* 2.3 Continuing common sequences: a harder example would be numbers at the start of lines (of arbitrary length)**

- **B\* 2.4 3-letter acronyms like “The Acrobatic Circus Group (ACG) and the Ringmaster Friendship Union (“ -> RFU**
  - GPT-2 Small is pretty good at this!

- **B\* 2.5 Converting names to emails, like “Katy Johnson <”->”katy_johnson”**
  - GPT-2 Small is pretty good at this!)
  - **C** An extension task is e.g. constructing an email from a snippet like the following:

  Name: Jess Smith
  
  Email: last name dot first name k @ gmail

- **C\* 2.6 - Interpret factual recall.**
  - Heavily crib off of the ROME paper’s work with causal tracing here, but how much more specific can you get? Can you find specific heads, or ideally specific neurons?

- **B\* 2.7 - Learning that words after full stops are capital letters**
  - NN predicts a “begins with a capital letter” direction in the residual stream that the model uses

- **B-C\* 2.8 - Counting objects described in text.**
  - E.g.: I picked up an apple, a pear, and an orange. I was holding three fruits.

- **C\* 2.9 - Interpreting memorisation.**
  - E.g., there are times when GPT-2 knows surprising facts like people’s contact information. How does that happen?

### Circuits in code models

**Note**: Try out GPT-Neo, Pythia or my SoLU models, which were trained on code

- **B\* 2.13 - Closing brackets**
  - Bonus: Tracking the right kinds of brackets - [, (, {, < etc

- **B\* 2.14 - Closing HTML tags**

### Confusing model observations

- **B\* 2.28 - Why is GPT-2 Small’s performance is ruined if you ablate MLP0?**

### Studying larger models

- **B-C\* 2.32 - GPT-J contains translation heads. Can you interpret how they work and what they do?**

## Techniques, Tooling and Automation

- **A\* 6.21 - Automated ways to analyse attention patterns to find Translation heads**

- **B\* 6.25 - The heads used in factual recall to move information about the fact to the final token**
  - I’d identify these via activation patching

### Using LLMs to interpret models - I don’t have great ideas here, but I’m sure there’s something!

- **B\* 6.40 - Can you use GPT-3 to generate counterfactual prompts with lined up tokens to do activation patching on novel problems?**
  - E.g., `“..., John gave a bottle of milk to -> Mary”` vs `“..., Mary gave a bottle of milk to -> John”` for the IOI task)

## Studying Learned Features in Language Models

- Lots of cool stuff here, but perhaps best to leave for the future at this stage.