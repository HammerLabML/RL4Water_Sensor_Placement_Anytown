from epyt_flow.simulation import ModelUncertainty
from epyt_flow.uncertainty import PercentageDeviationUncertainty

pattern_unc = PercentageDeviationUncertainty(0.05)
model_unc = ModelUncertainty(demand_pattern_uncertainty=pattern_unc)
uncertainty_file = '../../Data/Model_Uncertainties/uncertainty_1.json'
with open(uncertainty_file, 'w') as fp:
    fp.writelines(model_unc.to_json())
with open(uncertainty_file, 'r') as fp:
    content = fp.read()
    model_uncertainty = ModelUncertainty.load_from_json(content)
