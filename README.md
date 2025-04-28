CruiseFetchLITE (CFlite): ML-Based Memory Prefetcher

CruiseFetchLITE is an intelligent, ML-driven memory prefetcher designed for the ChampSim simulator framework. This prefetcher combines neural network techniques with behavioral clustering to accurately predict memory access patterns and improve system performance.

Key Elements:
The core file of this model is model.py and generate_moudule.py(using model to make prefetch for next test step) . The config files of model training are in the folder :config. 
Auto exe using autoRunCFlite.sh

For more Running instructions, please refer:README.CruiseFetch.md

Features

Rescource frienly :)

Even comsumer level PC can train the CFlite prefetcher model easily.

Integration with ChampSim

CruiseFetchLITE is designed to work seamlessly with the ChampSim simulation framework:

Load Trace Compatible: Works with ChampSim load traces containing instruction IDs, cycle counts, memory addresses, and hit/miss information

Competition Framework Ready: Follows the ML prefetching competition specifications for training and inference

Dual-Mode Operation: Supports both training on historical data and real-time inference

Research Foundation and citation:

CruiseFetchLITE builds on principles from several state-of-the-art prefetching project and research paper:

1.(Machine learning version ChampSim Prefetcher tester)https://github.com/Quangmire/ChampSim 

2.(2021 SOA Machine learning Prefetcher) Voyager https://github.com/Quangmire/voyager

3.Research Paper of Behavior Clustor in Machine learning prefetcher(2024): Duong, Quang, Akanksha Jain, and Calvin Lin. "A New Formulation of Neural Data Prefetching." 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). IEEE, 2024.


Technical Implementation

The core implementation is contained in model.py, which defines the CruiseFetchLITEModel class that inherits from the MLPrefetchModel base class. The model implements:

Training Pipeline: Processes historical access patterns to train the neural network

Inference Pipeline: Generates prefetch candidates based on current memory accesses

State Management: Maintains necessary state information for accurate predictions

For detailed implementation information, refer to the code documentation in model.py.

Citation

If you use CruiseFetchLITE in your research, please cite it as:

CruiseFetchLITE: A Lightweight Neural Prefetcher with Behavioral Clustering
