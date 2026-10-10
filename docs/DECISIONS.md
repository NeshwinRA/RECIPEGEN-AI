# Project decisions

## F05 request time semantics

`RecipeRequest.max_cooking_time` is the maximum requested **total recipe time**.
Compare it only with `Recipe.total_time_min`. The Food.com source's `minutes`
field is total time; do not infer or derive `prep_time_min` or `cook_time_min`
from it. Unknown times remain `null`.

## F09 vegetarian policy

For this project, `Vegetarian` means egg-free vegetarian. A recipe whose listed
ingredients match the egg keyword list does not pass vegetarian response
validation. `Eggetarian` allows egg but still checks for meat and fish terms.
`Vegan` checks for meat, fish, eggs, dairy, and honey terms.

These explicit ingredient keyword checks are a first safeguard, not proof of
dietary suitability and never an allergy-safety guarantee. Missing or ambiguous
ingredient information cannot establish that a recipe is suitable.
