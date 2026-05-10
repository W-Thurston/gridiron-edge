import yaml
from argparse import ArgumentParser

from src.model_pipelines.PFR.pfr_model_builder import pfr_model_builder

def main():

    print()
    print("Data Pipeline for Pro Football Reference Scraping and Feature Generation")
    ap = ArgumentParser()
    # ap.add_argument("--config_path", help="Path to yaml config file.")
    ap.add_argument("--ELO_only", nargs='+', type=str, help="A year: '2023-2024' & A week: 1 ")
    ap.add_argument("--eval_elo")
    ap.add_argument("--ranks_and_betting", nargs='+', type=str, help="A year: '2023-2024' & A week: 1 ")

    args = ap.parse_args()

    # with open(args.config_path) as file:
    #     config_dict = yaml.load(file, Loader=yaml.FullLoader)

    ## Produce Elo based recommendations ##
    if args.ELO_only:
        print("> Producing ELO only Recommendations -- START")
        ## Instantiate PFR Model Builder
        PFR_Model_Builder_object = pfr_model_builder()
        
        ## Produce Elo based predictions
        PFR_Model_Builder_object.predict_elo_only( year = args.ELO_only[0], week = int(args.ELO_only[1]))
        print("> Producing ELO only Recommendations -- END")
        print()

    ## Collect Data ##
    if args.eval_elo:
        print("> Evaluating ELO only Recommendations -- START")
        ## Instantiate PFR Model Builder
        PFR_Model_Builder_object = pfr_model_builder()
        
        ## Evaluate Elo based predictions
        print("> By Year:")
        PFR_Model_Builder_object.evaluate_elo_only(time_period='YEAR')
        print("> By Week:")
        PFR_Model_Builder_object.evaluate_elo_only(time_period='WEEK')
        print("> Evaluating ELO only Recommendations -- END")
        print()

    ## Collect Data ##
    if args.ranks_and_betting:
        print("> Building Ranks and Betting Excel -- START")
        ## Instantiate ranks_and_betting_builder object
        Ranks_and_Betting_builder_object = pfr_model_builder()
        
        ## Build Excel Visualization for Current week elo changes
        Ranks_and_betting_builder_object.build_new_ranks(year = args.ranks_and_betting[0], week = int(args.ranks_and_betting[1]))
        print("> Building Ranks and Betting Excel -- END")
        print()


if __name__ == '__main__':
    """
    "Usage:"
    python PFR_model_pipeline_run.py
        --ELO_only '2023-2024' int(curr_wk_#) 
        --eval_elo 0
        --ranks_and_betting '2023-2024' int(last_wk_#) 
    """
    main()
