#!/usr/bin/env bash
# GURU dataset splits and reward-service requirements.

configure_data() {
    TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-${REPO_ROOT}/data/train}
    VAL_DATA_DIR=${VAL_DATA_DIR:-${REPO_ROOT}/data/online_eval}

    MATH_TRAIN_FILES=(
        "${TRAIN_DATA_DIR}/math__combined_54.4k.parquet"
    )
    CODE_TRAIN_FILES=(
        "${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k.parquet"
        "${TRAIN_DATA_DIR}/codegen__primeintellect_7.5k.parquet"
        "${TRAIN_DATA_DIR}/codegen__taco_8.8k.parquet"
    )
    FULL_TRAIN_FILES=(
        "${CODE_TRAIN_FILES[@]}"
        "${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet"
        "${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet"
        "${TRAIN_DATA_DIR}/logic__barc_1.6k.parquet"
        "${TRAIN_DATA_DIR}/logic__graph_logical_1.2k.parquet"
        "${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k.parquet"
        "${TRAIN_DATA_DIR}/logic__zebra_puzzle_1.3k.parquet"
        "${MATH_TRAIN_FILES[@]}"
        "${TRAIN_DATA_DIR}/simulation__codeio_3.7k.parquet"
        "${TRAIN_DATA_DIR}/stem__web_3.6k.parquet"
        "${TRAIN_DATA_DIR}/table__hitab_4.3k.parquet"
        "${TRAIN_DATA_DIR}/table__multihier_1.5k.parquet"
    )

    MATH_VAL_FILES=(
        "${VAL_DATA_DIR}/math__math_500.parquet"
        "${VAL_DATA_DIR}/math__aime_repeated_8x_240.parquet"
    )
    CODE_VAL_FILES=(
        "${VAL_DATA_DIR}/codegen__humaneval_164.parquet"
        "${VAL_DATA_DIR}/codegen__mbpp_200.parquet"
    )
    FULL_VAL_FILES=(
        "${CODE_VAL_FILES[@]}"
        "${VAL_DATA_DIR}/logic__arcagi1_200.parquet"
        "${VAL_DATA_DIR}/logic__ordering_puzzle_dataset_100.parquet"
        "${VAL_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet"
        "${MATH_VAL_FILES[@]}"
        "${VAL_DATA_DIR}/math__amc_repeated_4x_332.parquet"
        "${VAL_DATA_DIR}/simulation__codeio_200.parquet"
        "${VAL_DATA_DIR}/stem__supergpqa_200.parquet"
        "${VAL_DATA_DIR}/table__hitab_200.parquet"
        "${VAL_DATA_DIR}/table__multihier_200.parquet"
    )

    case "${TRAIN_SCOPE}" in
        full)
            TRAIN_FILES=("${FULL_TRAIN_FILES[@]}")
            VAL_FILES=("${FULL_VAL_FILES[@]}")
            NEED_SANDBOX=1
            NEED_STEM_VERIFIER=1
            ;;
        math)
            TRAIN_FILES=("${MATH_TRAIN_FILES[@]}")
            VAL_FILES=("${MATH_VAL_FILES[@]}")
            NEED_SANDBOX=0
            NEED_STEM_VERIFIER=0
            ;;
        code)
            TRAIN_FILES=("${CODE_TRAIN_FILES[@]}")
            VAL_FILES=("${CODE_VAL_FILES[@]}")
            NEED_SANDBOX=1
            NEED_STEM_VERIFIER=0
            ;;
        *)
            echo "TRAIN_SCOPE must be one of: full, math, code" >&2
            exit 2
            ;;
    esac
}
