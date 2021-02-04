from epidemioptim.environments.models.prague_ode_seirah_model import PragueOdeSeirahModel
from epidemioptim.environments.models.sqeir_model import SqeirModel


list_models = ['prague_seirah', 'sqeir']
def get_model(model_id, params={}):
    """
    Get the epidemiological model.

    Parameters
    ----------
    model_id: str
        Model identifier.
    params: dict
        Dictionary of experiment parameters.

    """
    assert model_id in list_models, "Model id should be in " + str(list_models)
    if model_id == 'prague_seirah':
        return PragueOdeSeirahModel(**params)
    elif model_id == 'sqeir':
        return SqeirModel(**params)
    else:
        raise NotImplementedError

#TODO: add tests for model registration #???


