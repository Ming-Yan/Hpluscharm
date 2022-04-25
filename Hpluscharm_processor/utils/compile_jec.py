import importlib.resources
import contextlib
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory

jec_name_map = {
    'JetPt': 'pt',
    'JetMass': 'mass',
    'JetEta': 'eta',
    'JetA': 'area',
    'ptGenJet': 'pt_gen',
    'ptRaw': 'pt_raw',
    'massRaw': 'mass_raw',
    'Rho': 'event_rho',
   }


def jet_factory_factory(files):
    ext = extractor()
    ext.add_weight_sets([f"* * {file}" for file in files])
    ext.finalize()
    jec_stack = JECStack(ext.make_evaluator())

    return CorrectedJetsFactory(jec_name_map, jec_stack)


jet_factory = {
    "UL16_MC": jet_factory_factory(
        files=[
        "data/JEC_JERSF/UL16/Summer20UL16_JRV3_MC_SF_AK4PFchs.jersf.txt",
        "data/JEC_JERSF/UL16/Summer20UL16_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
        "data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L1FastJet_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L2Relative_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L3Absolute_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16_V7_MC_L2L3Residual_AK4PFchs.jec.txt",
        ]
    ),
    "UL16_MCAPV": jet_factory_factory(
        files=[
        "data/JEC_JERSF/UL16/Summer20UL16APV_JRV3_MC_SF_AK4PFchs.jersf.txt",
        "data/JEC_JERSF/UL16/Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs.jr.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L1FastJet_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L2Relative_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L3Absolute_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_V7_MC_L2L3Residual_AK4PFchs.jec.txt",
        ]
    ),
    "Run2016BCD": jet_factory_factory(
        files=[
             "data/JEC_JERSF/UL16/Summer19UL16APV_RunBCD_V7_DATA_L1FastJet_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_RunBCD_V7_DATA_L2L3Residual_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_RunBCD_V7_DATA_L2Relative_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_RunBCD_V7_DATA_L3Absolute_AK4PFchs.jec.txt",
        ]
    ),
    "Run2016EF": jet_factory_factory(
        files=[
        "data/JEC_JERSF/UL16/Summer19UL16APV_RunEF_V7_DATA_L1FastJet_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_RunEF_V7_DATA_L2L3Residual_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_RunEF_V7_DATA_L2Residual_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16APV_RunEF_V7_DATA_L3Absolute_AK4PFchs.jec.txt",
        ]
    ),
    
    "Run2016FGH": jet_factory_factory(
        files=[
        "data/JEC_JERSF/UL16/Summer19UL16_RunFGH_V7_DATA_L1FastJet_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16_RunFGH_V7_DATA_L2L3Residual_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16_RunFGH_V7_DATA_L2Residual_AK4PFchs.jec.txt",
        "data/JEC_JERSF/UL16/Summer19UL16_RunFGH_V7_DATA_L3Absolute_AK4PFchs.jec.txt",
        ]
    ),
   
}




if __name__ == "__main__":
    import sys
    import gzip
    # jme stuff not pickleable in coffea
    import cloudpickle
    with gzip.open(sys.argv[-1], "wb") as fout:
        cloudpickle.dump(
            {
                "jet_factory": jet_factory,
            },
            fout
        )
