from .loss_func import NewLoss,PriceLoss,Weights,VolumeLoss,MSELoss,MAELoss
from .loss_func import WeightedPriceLoss,WeightedVolumeLoss, WeightedMSELoss
from .loss_func import TotalMoneyLoss
from .loss_func import CELoss, MAELoss
from .loss_func import WeightedMSELoss_with_reg,RegLoss
from .loss_func import AllLoss


from .masked_loss import MaskedPriceLoss, MaskedVolumeLoss, WeightedMaskedPriceLoss, WeightedMaskedVolumeLoss, MaskedMSELoss




metrics = {
    "mse_loss": MSELoss,
    "price_loss": PriceLoss,
    "volume_loss": VolumeLoss,
    "weighted_price_loss": WeightedPriceLoss,
    "weighted_volume_loss": WeightedVolumeLoss,
    "weighted_mse_loss": WeightedMSELoss,
    "total_money_loss": TotalMoneyLoss,
    "cross_entropy_loss": CELoss,
    "mae_loss": MAELoss,
    # "new_loss": NewLoss,
    "mae_loss": MAELoss,
    
    "masked_price_loss": MaskedPriceLoss,
    "masked_volume_loss": MaskedVolumeLoss,
    "weighted_masked_price_loss": WeightedMaskedPriceLoss,
    "weighted_masked_volume_loss": WeightedMaskedVolumeLoss,
    "weighted_mse_loss_with_reg": WeightedMSELoss_with_reg,
    "reg_loss": RegLoss,
    "all_loss": AllLoss,
    "masked_mse_loss": MaskedMSELoss
}

__all__ = ['NewLoss','PriceLoss','Weights','WeightedPriceLoss','VolumeLoss',
           'WeightedVolumeLoss','MSELoss','WeightedMSELoss',
           'TotalMoneyLoss',
           'CELoss','MAELoss',
           'metrics',
           
           'MaskedPriceLoss','MaskedVolumeLoss',
           'WeightedMaskedPriceLoss','WeightedMaskedVolumeLoss'
           'WeightedMSELoss_with_reg','RegLoss','AllLoss',"MAELoss",
           ]