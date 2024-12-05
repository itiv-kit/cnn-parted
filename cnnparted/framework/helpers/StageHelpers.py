
STAGE_DEPENDENCIES = {}

def register_required_stage(required_stages):
    def wrapper(cls):
        STAGE_DEPENDENCIES[cls] = required_stages
        return cls
    
    return wrapper