# FarmCheck AI — Dataset & Annotation Report

## Overview
- **Total images:** 1500
- **Classes:** 15
- **Indicators:** 6
- **Compliant (healthy):** 300
- **Non-compliant (diseased):** 1200
- **Image size (preprocessed):** 224×224px

## Split Distribution
| split   |   filename |
|:--------|-----------:|
| test    |        225 |
| train   |       1049 |
| val     |        226 |

## Indicator Distribution
| indicator           |   filename |
|:--------------------|-----------:|
| bacterial_infection |        200 |
| crop_healthy        |        300 |
| fungal_blight       |        500 |
| leaf_disease        |        200 |
| pest_infestation    |        100 |
| viral_infection     |        200 |

## Class → Indicator Mapping
| class                                       | indicator           | compliant   |   binary |
|:--------------------------------------------|:--------------------|:------------|---------:|
| Pepper__bell___Bacterial_spot               | bacterial_infection | False       |        0 |
| Pepper__bell___healthy                      | crop_healthy        | True        |        1 |
| Potato___Early_blight                       | fungal_blight       | False       |        0 |
| Potato___Late_blight                        | fungal_blight       | False       |        0 |
| Potato___healthy                            | crop_healthy        | True        |        1 |
| Tomato_Bacterial_spot                       | bacterial_infection | False       |        0 |
| Tomato_Early_blight                         | fungal_blight       | False       |        0 |
| Tomato_Late_blight                          | fungal_blight       | False       |        0 |
| Tomato_Leaf_Mold                            | fungal_blight       | False       |        0 |
| Tomato_Septoria_leaf_spot                   | leaf_disease        | False       |        0 |
| Tomato_Spider_mites_Two_spotted_spider_mite | pest_infestation    | False       |        0 |
| Tomato__Target_Spot                         | leaf_disease        | False       |        0 |
| Tomato__Tomato_YellowLeaf__Curl_Virus       | viral_infection     | False       |        0 |
| Tomato__Tomato_mosaic_virus                 | viral_infection     | False       |        0 |
| Tomato_healthy                              | crop_healthy        | True        |        1 |

## Notes
- Stratified sampling used to ensure class balance across splits.
- All images resized to 224×224 and converted to RGB.
- Labels auto-generated from folder names and verified in Label Studio.
- Binary label: 1 = compliant (healthy), 0 = non-compliant (diseased).