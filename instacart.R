# data table
library(data.table)

# A total of 134 products
aisles <- fread("aisles.csv")
# 21 departments
departments <- fread("departments.csv")
# prior data of all the users
order_products_prior <- fread("order_products__prior.csv")
# training data of all the users
order_products_train <- fread("order_products__train.csv")
# orders
orders <- fread("orders.csv")
# products
products <- fread("products.csv")

# Converting to factor
aisles$aisle <- as.factor(aisles$aisle)
departments$department <- as.factor(departments$department)
orders$eval_set <- as.factor(orders$eval_set)
products$product_name <- as.factor(products$product_name)

# reference - https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/34727

# The important datasets are:
# order.csv - It contains all the information regarding an order like
#             order id, user id, evaluation set it belongs to,
#             day of the week it was ordered(order_dow), hour,
#             and days since prior order.
# order_product__prior.csv - it contains order_id and corresponding product_id,
#                           product order in the cart, and whether it is ordered 
#                           again or not. It has prior data of all the users.
# order_product_train.csv - same as prior but it has only training data.

# Creating a product level by combing product, department, aisles
product_level <- merge(x = products, y = aisles, by = "aisle_id")
product_level <- merge(x = product_level, y = departments, by = "department_id")
product_level$department_id <- NULL
product_level$aisle_id <- NULL
product_level <- arrange(product_level, product_id)

# removing unused data
rm(aisles,departments, products)
gc()

# !! Remember to take only order id in training set that are also in orders

# combining order data and prior data
ordered_products <- merge(x = orders, y = order_products_prior, by = "order_id")
rm(order_products_prior)
gc()


library(dplyr)

# product reorder probability and avg_cart_position of each cart
product_probab <- ordered_products %>% arrange(user_id, order_number, product_id) %>% group_by(product_id) %>% 
  summarise(product_orders = n(), product_reorders = sum(reordered), avg_cart_pos = mean(add_to_cart_order))
product_probab$reorder_probab <- product_probab$product_reorders/product_probab$product_orders
product_probab$product_reorders <- NULL

# calculating user buy probab

# calculating user order probability by looking at its prior orders 
users_probab <- orders %>% filter(eval_set == "prior") %>% group_by(user_id) %>% 
  summarise(user_orders = max(order_number), user_period = sum(days_since_prior_order, na.rm = TRUE),
    avg_days_since_prior = mean(days_since_prior_order, na.rm = TRUE))

# calculating total_products, reorder probability and num of products
users_reorder_probab <- ordered_products %>% group_by(user_id) %>%
  summarise(user_total_products = n(), user_reorder_probab = sum(reordered == 1) / sum(order_number > 1),
    num_products = n_distinct(product_id))

# merging above two to get user level
users_probab <- merge(x = users_probab, y = users_reorder_probab, by = "user_id", all.x = TRUE)

# filtering training and testing data from orders
train_test <- orders %>% filter(eval_set != "prior") %>% select(user_id, order_id, eval_set, days_since_prior_order)

# left join users_probab with train_test data
users_probab <- merge(x = users_probab, y = train_test, all.x = TRUE)

rm(train_test)


# calculating average cart position and total orders of each product user purchased
user_product_cart <- ordered_products %>% group_by(user_id, product_id) %>% 
  summarise( user_product_orders = n(), avg_user_product_pos = mean(add_to_cart_order))


rm(users_reorder_probab, product_level,ordered_products)
gc()


# now combining all the user_level and product _level info
user_product_cart <- merge(user_product_cart,product_probab, by = "product_id", all.x = TRUE)
user_product_cart <- merge(user_product_cart,users_probab, by = "user_id", all.x = TRUE)


#taking only user_ids that are common in orders
order_products_train$user_id <- orders$user_id[match(order_products_train$order_id, orders$order_id)]
# combining training data and infered data by product id and user id
order_products_train <- order_products_train %>% select(user_id, product_id, reordered)
user_product_cart <- merge(user_product_cart, order_products_train, by = c("user_id", "product_id"), all.x = TRUE)


rm(product_probab, order_products_train, users_probab, orders)
gc()

# training data
train <- as.data.frame(user_product_cart[user_product_cart$eval_set == "train",])
# removing char - xgboost
train$eval_set <- NULL
#no need
train$user_id <- NULL
train$product_id <- NULL
train$order_id <- NULL
train$reordered[is.na(train$reordered)] <- 0

# testing data
test <- as.data.frame(user_product_cart[user_product_cart$eval_set == "test",])
test$eval_set <- NULL
test$user_id <- NULL
test$reordered <- NULL

rm(user_product_cart)
gc()

#we got our training and testing data
# we have taken only numeric data because we will be using xgboost

# xgboost- Gradient Boosting
library(xgboost)

# refrence - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
params <- list(
  # logistic model
  "objective"           = "reg:logistic",
  
  # logless for cross-validation
  "eval_metric"         = "logloss", 
  
  #learning rate
  "eta"                 = 0.1,
  
  #depth of tree
  "max_depth"           = 6, 
  
  # min sum of weights
  # should be high enough to prevent over fitting
  # but not too high for over fitting
  "min_child_weight"    = 10,
  
  # the min loss value require to split
  "gamma"               = 0.70,
  
  # fraction of observations to be included in each tree 
  # generally varies from 0.5-1
  "subsample"           = 0.75,
  
  # fraction of column to be randomly sample in each tree
  "colsample_bytree"    = 0.95,
  
  # regularization coefficients
  "alpha"               = 2e-05,
  "lambda"              = 10 
)

# taking 1% of data
subtrain <- train %>% sample_frac(0.1)
X <- xgb.DMatrix(as.matrix(subtrain %>% select(-reordered)), label = subtrain$reordered)
model <- xgboost(data = X, params = params, nrounds = 80)

rm(X, subtrain)
gc()

# predicting reordered values from test dataset
X <- xgb.DMatrix(as.matrix(test %>% select(-order_id, -product_id)))
test$reordered <- predict(model, X)

test$reordered <- (test$reordered > 0.21) * 1

# summarise as order_id and products
submission <- test %>% filter(reordered == 1) %>% group_by(order_id) %>%
  summarise(products = paste(product_id, collapse = " "))

# filliing the missing values
missing <- data.frame(
  order_id = unique(test$order_id[!test$order_id %in% submission$order_id]), products = "None")
submission <- submission %>% bind_rows(missing) %>% arrange(order_id)

write.csv(submission, file = "submission.csv", row.names = F)
