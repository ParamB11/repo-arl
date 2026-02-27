import argparse

def get_config():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-carl_env_name', type=str, help='name of the CARL environment', default='CARLPendulum')
    # parser.add_argument('-context_labels', '--context_labels', nargs='+', type=str, help='context labels', required=True)
    parser.add_argument('-context_labels', '--context_labels', nargs='+', type=str, help='context labels', default='')
    parser.add_argument('-prefix_experts', type=str, help='prefix of the saved experts', default='')
    parser.add_argument('-multiplier', '--multiplier', type=int, nargs='+', help='context multiplier for saving the models', default=0)
    parser.add_argument('-up_policy_path', type=str, help='path to dir where up policy is saved', default='data/saved_models/')
    parser.add_argument('-up_prefix', type=str, help='prefix used to load up policy', required=True)
    parser.add_argument('-up_suffix', type=str, help='suffix used to load up policy', default='')
    parser.add_argument('-savedir', type=str, help='directory to save models', default='saved_models/')
    parser.add_argument('-device', type=str, help='device to use for training', default='cpu')

    return parser