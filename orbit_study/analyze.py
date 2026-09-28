"""CLI для явного повторного анализа сохранённого набора орбит."""
from __future__ import annotations

import argparse

from .user_run import run_selected_case


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n",type=int,required=True,choices=range(2,11))
    parser.add_argument("--horizon-days",type=float,required=True)
    parser.add_argument("--threshold-deg",type=float,choices=(1.,2.),default=2.)
    parser.add_argument("--output-dir")
    args=parser.parse_args()
    result=run_selected_case(args.n,args.threshold_deg,args.horizon_days,args.output_dir)
    print(f"{result['folder']}: {result['status']}; пролётов "
          f"{result['qualifying_passes']}; min "
          f"{result['closest_pair']['distance_km']:.6f} км "
          f"({result['closest_pair']['pair']})")


if __name__=="__main__":
    main()
