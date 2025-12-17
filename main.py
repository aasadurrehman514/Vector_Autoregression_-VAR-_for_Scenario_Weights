import argparse
from datetime import datetime

from VAR_weights import cal_senario_weights
from visuals import plot_macro_distributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--WEO_Data', type=str, default=r'inputs\WEO_Data.csv')
    parser.add_argument('--save_csv', type=str, default='yes')
    parser.add_argument('--visuals', type=str, default='no')


    start_time = datetime.now()
    args = parser.parse_args()
    event = vars(args)

    sim_df,scenario_weights = cal_senario_weights(event)

    if event['visuals'] == 'yes':
        plot_macro_distributions(sim_df,scenario_weights.to_dict())

    end_time = datetime.now()
    print("\n** Total Elapsed Runtime:", str(end_time - start_time))
    
if __name__ == "__main__":
    main()
