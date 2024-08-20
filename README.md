# RouteMaker
Used to solve MDMTSP-CFA, MDMTSP-CFD, SDMTSP

SDMTSP: Single-depot multiple traveling salesman problem, agents must start and end traveling at the depot

MDMTSP: Dedicated multi-depot MTSP, agent must start traveling at their dedicated depots. 

MDMTSP-CFD: one MDMTSP variation, in which agents start and commence travels from their dedicated depots

MDMTSP-CFA: one MDMTSP variation, in which agents start traveling from any locations.

## Dependencies

## Training
### Training RouteMaker for MDMTSP-CFA
```python
python train.py --problem cfa
```

### Training RouteMaker for MDMTSP-CFD
```python
python train.py --problem cfd
```

### Training RouteMaker for SDMTSP
```python
python train.py --problem sdmtsp
```

## Testing
### Testing RouteMaker tailored for MDMTSP-CFA on testset with 10 agents and 100 locations. Code will automatically load the learned model under "savemodel/"
```python
python eval --problem cfa --anum 10 --cnum 100
```

### Testing RouteMaker tailored for MDMTSP-CFD on testset with 20 agents and 200 locations. Code will automatically load the learned model under "savemodel/"
```python
python eval --problem cfd --anum 20 --cnum 200
```

### Testing RouteMaker tailored for SDMSTP on testset with 20 agents and 200 locations. Code will automatically load the learned model under "savemodel/"
```python
python eval --problem sdmtsp --anum 20 --cnum 200
```

## Note
The number of depots and current positions are not inclued in the number of locations.

Use ulimit -n 99999 to enable high parallel before running the code

In the rarely possibility that RouteMaker cannot converge, please delete the bad model and retrain a new model. 

