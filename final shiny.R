# Load required libraries
library(tidyr)
library(xgboost)
library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(ggplot2)
library(corrplot)
library(inspectdf)

# UI
ui <- dashboardPage(
  dashboardHeader(title = "Sales Forecasting Shiny App"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Upload Data", tabName = "upload", icon = icon("cloud-upload-alt")),
      menuItem("Summary", tabName = "summary", icon = icon("table")),
      menuItem("Exploratory Data Analysis", tabName = "data_viz", icon = icon("chart-bar")),
      menuItem("Model Evaluation Metrics", tabName = "evaluation", icon = icon("chart-line")),
      menuItem("Feature Importance", tabName = "importance", icon = icon("chart-pie")),
      menuItem("Forecasting", tabName = "predictions", icon = icon("line-chart"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "upload",
              fluidRow(
                column(6,
                       fileInput("file", "Upload CSV File"),
                       br(),
                       actionButton("process", "Process Data", class = "btn btn-primary"),
                       br(),
                       h4("Data Processing in Progress...", id = "processing_text", style = "color: #ff0000; display: none;"),
                       h4("Data Successfully Processed!", id = "processed_text", style = "color: #00ff00; display: none;")
                ),
                column(6,
                       h4("Model Information:", style = "font-weight:bold;"),
                       HTML("This model was built with the help of <span style='color: blue;'>XGBoost</span> algorithm, providing accurate sales predictions.")
                )
              ),
              fluidRow(
                column(6,
                       h4("Correlation Map", style = "font-weight:bold;"),
                       plotOutput("correlation_map")
                ),
                column(6,
                       h4("Missing Values Visualization", style = "font-weight:bold;"),
                       plotOutput("missing_values_plot")
                )
              )
      ),
      tabItem(tabName = "summary",
              DTOutput("data_summary_table")
      ),
      tabItem(tabName = "data_viz",
              fluidRow(
                column(8,
                       selectInput("data_viz_plot", "Select Plot Type:", 
                                   choices = c("Box Plot", "Histogram", "Bar Plot", "Heatmap"),
                                   selected = "Box Plot"),
                       plotlyOutput("data_viz_output")
                )
                
              ),
              fluidRow(
                column(8,
                       h4("Histograms of Numerical Variables", style = "font-weight:bold;"),
                       plotOutput("numerical_histograms")
                )
              )
      ),
      tabItem(tabName = "evaluation",
              fluidRow(
                column(12,
                       h3(icon("line-chart"), "Model Evaluation Metrics", style = "color: #0000ff;"),
                       tableOutput("model_evaluation_table")
                )
              )
      ),
      tabItem(tabName = "importance",
              fluidRow(
                column(12,
                       h3(icon("bar-chart-o"), "Feature Importance", style = "color: #0000ff;"),
                       plotlyOutput("feature_importance_table")
                )
              )
      ),
      tabItem(tabName = "predictions",
              fluidRow(
                column(8,
                       dateRangeInput("date_range", "Select a Date Range:", 
                                      start = "2010-05-02", end = "2012-09-28",
                                      min = "2010-05-02", max = "2012-09-28"),
                       selectInput("prediction_type", "Select Prediction Type:", 
                                   choices = c("Predicted Sales"),
                                   selected = "Predicted Sales"),
                       plotlyOutput("predictions_plot"),
                       downloadButton("download_predictions", "Download Predictions", class = "btn btn-success")
                )
              )
      )
    )
  )
)

# Server
server <- function(input, output) {
  options(shiny.maxRequestSize = 50 * 1024^2) 
  
  processed_data <- reactiveVal(NULL)
  model_importance <- reactiveVal(NULL)  # Store the feature importance data
  
  observeEvent(input$process, {
    req(input$file)
    
    data <- read.csv(input$file$datapath)
    
    # Data processing (you might need to modify this based on your data structure)
    data$Date <- as.Date(data$Date, format = "%m/%d/%Y")
    
    # Include the "IsHoliday.y" column in the training and testing datasets
    x_train <- data.matrix(data[, !(names(data) %in% c("Weekly_Sales", "IsHoliday.y"))])
    y_train <- data$Weekly_Sales
    x_test <- data.matrix(data[, !(names(data) %in% c("Weekly_Sales", "IsHoliday.y"))])
    
    processed_data(data)
    
    # Assuming you have already defined and trained your XGBoost model
    # Modify the xgboost parameters (e.g., nrounds, max_depth, etc.) based on your requirement
    xgb_model <- xgboost(data = x_train, label = y_train, nrounds = 100, nthread = 4)
    
    # Calculate feature importance using the xgboost built-in function
    importance <- xgb.importance(model = xgb_model)
    model_importance(importance)
    
    output$processed_text <- renderText({
      "Data Successfully Processed!"
    })
  })
  
  # Server - Missing Values Visualization
  output$missing_values_plot <- renderPlot({
    df <- processed_data()
    if (!is.null(df)) {
      missing_values <- colSums(is.na(df))
      missing_df <- data.frame(Columns = names(missing_values), Missing_Count = missing_values)
      
      ggplot(missing_df, aes(x = reorder(Columns, -Missing_Count), y = Missing_Count)) +
        geom_bar(stat = "identity", fill = "#1f78b4") +
        labs(title = "Missing Values Visualization", x = "Columns", y = "Missing Count") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }
  })
  
  output$data_summary_table <- renderDT({
    data <- processed_data()
    if (!is.null(data)) {
      datatable(data, options = list(pageLength = 10, scrollX = TRUE))
    }
  })
  
  # Server - Continue with the data visualization plots in "Exploratory Data Analysis" tab
  output$data_viz_output <- renderPlotly({
    df <- processed_data()
    if (!is.null(df)) {
      req(input$data_viz_plot)
      
      plot_data <- df  # Use the entire data for visualization (you might want to modify this based on your plot type)
      
      p <- NULL
      
      if (input$data_viz_plot == "Box Plot") {
        p <- ggplot(plot_data, aes(x = Type, y = Weekly_Sales, fill = Type)) +
          geom_boxplot() +
          labs(title = "Box Plot of Weekly Sales by Type", x = "Type", y = "Weekly Sales")
      } else if (input$data_viz_plot == "Histogram") {
        p <- ggplot(plot_data, aes(x = Weekly_Sales, fill = Type)) +
          geom_histogram(binwidth = 2000) +
          labs(title = "Histogram of Weekly Sales", x = "Weekly Sales", y = "Count")
      } else if (input$data_viz_plot == "Bar Plot") {
        # Calculate total sales for each type
        total_sales_data <- df %>%
          group_by(Type) %>%
          summarise(Total_Sales = sum(Weekly_Sales, na.rm = TRUE))
        
        p <- ggplot(total_sales_data, aes(x = reorder(Type, Total_Sales), y = Total_Sales)) +
          geom_bar(stat = "identity", fill = "#1f78b4") +
          labs(title = "Total Sales by Type", x = "Type", y = "Total Sales") +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
      } else if (input$data_viz_plot == "Heatmap") {
        # For the heatmap, you need to map 'Type' to the x-axis and 'Size' to the y-axis
        p <- ggplot(plot_data, aes(x = Type, y = Size, fill = Weekly_Sales)) +
          geom_tile() +
          labs(title = "Heatmap of Weekly Sales by Type and Size", x = "Type", y = "Size", fill = "Weekly Sales")
      }
      
      ggplotly(p, tooltip = c("x", "y", "fill"), dynamicTicks = TRUE)
    }
  })
  
  # Server - Plot histograms of numerical variables
  # Server - Plot histograms of numerical variables
  output$numerical_histograms <- renderPlot({
    df <- processed_data()
    if (!is.null(df)) {
      numerical_cols <- c("Size", "Temperature", "Fuel_Price", "CPI", "Unemployment")
      
      hist_plots <- list()
      for (col in numerical_cols) {
        p <- ggplot(df, aes(x = !!sym(col))) +
          geom_histogram(fill = "steelblue", color = "white") +
          labs(title = paste("Histogram of", col)) +
          theme(plot.title = element_text(size = 16),  # Increase the title size
                axis.text = element_text(size = 14),   # Increase axis label size
                axis.title = element_text(size = 14))  # Increase axis title size
        
        hist_plots[[col]] <- p
      }
      
      gridExtra::grid.arrange(grobs = hist_plots, ncol = 2, widths = c(10, 10))  # Increase widths for larger plots
    }
  })
  
  
  
  # Server - Calculate evaluation metrics
  output$model_evaluation_table <- renderTable({
    df <- processed_data()
    if (!is.null(df)) {
      req(input$date_range)
      
      date_start <- input$date_range[1]
      date_end <- input$date_range[2]
      
      # Filter data for the selected date range
      eval_data <- df[df$Date >= as.Date(date_start) & df$Date <= as.Date(date_end), ]
      train_data <- df[df$Date < as.Date(date_start), ]
      test_data <- df[df$Date >= as.Date(date_start) & df$Date <= as.Date(date_end), ]
      
      # Calculate evaluation metrics for all data (training data)
      mse_all <- mean((train_data$Weekly_Sales - train_data$Predicted_Sales)^2)
      rmse_all <- sqrt(mse_all)
      mae_all <- mean(abs(train_data$Weekly_Sales - train_data$Predicted_Sales))
      r_squared_all <- 1 - sum((train_data$Weekly_Sales - train_data$Predicted_Sales)^2) / sum((train_data$Weekly_Sales - mean(train_data$Weekly_Sales))^2)
      
      # Calculate evaluation metrics for holidays (test data - holidays)
      test_data_holiday <- test_data[test_data$IsHoliday.y == TRUE, ]
      mse_holiday <- mean((test_data_holiday$Weekly_Sales - test_data_holiday$Predicted_Sales)^2)
      rmse_holiday <- sqrt(mse_holiday)
      mae_holiday <- mean(abs(test_data_holiday$Weekly_Sales - test_data_holiday$Predicted_Sales))
      r_squared_holiday <- 1 - sum((test_data_holiday$Weekly_Sales - test_data_holiday$Predicted_Sales)^2) / sum((test_data_holiday$Weekly_Sales - mean(test_data_holiday$Weekly_Sales))^2)
      
      # Calculate evaluation metrics for non-holidays (test data - non-holidays)
      test_data_non_holiday <- test_data[test_data$IsHoliday.y == FALSE, ]
      mse_non_holiday <- mean((test_data_non_holiday$Weekly_Sales - test_data_non_holiday$Predicted_Sales)^2)
      rmse_non_holiday <- sqrt(mse_non_holiday)
      mae_non_holiday <- mean(abs(test_data_non_holiday$Weekly_Sales - test_data_non_holiday$Predicted_Sales))
      r_squared_non_holiday <- 1 - sum((test_data_non_holiday$Weekly_Sales - test_data_non_holiday$Predicted_Sales)^2) / sum((test_data_non_holiday$Weekly_Sales - mean(test_data_non_holiday$Weekly_Sales))^2)
      
      metrics_df <- data.frame(
        Metric = c("Training Data (All)", "Test Data", "Test Data"),
        MSE = c(mse_all, mse_holiday, mse_non_holiday),
        RMSE = c(rmse_all, rmse_holiday, rmse_non_holiday),
        MAE = c(mae_all, mae_holiday, mae_non_holiday),
        R_squared = c(r_squared_all, r_squared_holiday, r_squared_non_holiday)
      )
      
      metrics_df
    }
  })
  
  # Server - Display feature importance as a bar chart
  output$feature_importance_table <- renderPlotly({
    importance <- model_importance()
    if (!is.null(importance)) {
      importance_df <- as.data.frame(importance) %>%
        arrange(desc(Gain))
      
      p <- ggplot(importance_df, aes(x = reorder(Feature, -Gain), y = Gain)) +
        geom_bar(stat = "identity", fill = "#1f78b4") +
        labs(title = "Feature Importance", x = "Feature", y = "Gain") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      ggplotly(p)
    }
  })
  
  # Server - Forecasting predictions
  output$predictions_plot <- renderPlotly({
    df <- processed_data()
    if (!is.null(df)) {
      req(input$date_range)
      req(input$prediction_type)
      
      date_start <- input$date_range[1]
      date_end <- input$date_range[2]
      
      plot_data <- df[df$Date >= as.Date(date_start) & df$Date <= as.Date(date_end), ]
      
      if (nrow(plot_data) == 0) {
        return(shiny::tags$h3("No Data Available for the Selected Date Range", style = "color: #ff0000;"))
      } else {
        # Convert IsHoliday.x to factor to avoid 'Continuous value supplied to discrete scale' error
        plot_data$IsHoliday.x <- factor(plot_data$IsHoliday.x, levels = c(FALSE, TRUE))
        
        p <- ggplot(plot_data, aes(x = Date)) +
          labs(title = "Sales Predictions", x = "Date", y = "Sales", color = "Sales Type") +
          theme_minimal() +
          theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 2)) +
          guides(color = guide_legend(override.aes = list(linetype = c(1, 1))))  # Set linetype for legend labels
        
        if (input$prediction_type == "Predicted Sales") {
          # Calculate weekly sales by summing the Predicted_Sales for each week
          predicted_sales_data <- df %>%
            filter(Date >= as.Date(date_start) & Date <= as.Date(date_end)) %>%
            group_by(week = format(Date, "%Y-%U")) %>%
            summarise(Predicted_Sales = sum(Predicted_Sales, na.rm = TRUE))
          
          # Convert the week back to Date format to use in the plot
          predicted_sales_data$Date <- as.Date(paste0(predicted_sales_data$week, "-1"), format = "%Y-%U-%u")
          
          # Create a separate data frame for actual sales in the selected date range
          actual_sales_data <- df %>%
            filter(Date >= as.Date(date_start) & Date <= as.Date(date_end)) %>%
            group_by(week = format(Date, "%Y-%U")) %>%
            summarise(Weekly_Sales = sum(Weekly_Sales, na.rm = TRUE))
          
          # Convert the week back to Date format to use in the plot
          actual_sales_data$Date <- as.Date(paste0(actual_sales_data$week, "-1"), format = "%Y-%U-%u")
          
          # Add lines for predicted sales and actual sales
          p <- p + geom_line(data = predicted_sales_data, aes(y = Predicted_Sales),
                             linetype = "dotted", color = "black", size = 1) +
            geom_line(data = actual_sales_data, aes(y = Weekly_Sales),
                      color = "blue", size = 1)
        }
        
        ggplotly(p, tooltip = c("x", "y", "color"), dynamicTicks = TRUE, height = 400)  # Set the height of the plot to 600 pixels
      }
    }
  })
  
  output$download_predictions <- downloadHandler(
    filename = function() {
      "final_data_with_predictions.csv"
    },
    content = function(file) {
      # Implement the code to obtain the predicted sales data and store it in 'predicted_sales_data'
      # Assuming you have a column named "Predicted_Sales" in your data frame with the predictions
      predicted_sales_data <- subset(final_data_with_predictions, Date >= as.Date("2010-05-02") & Date <= as.Date("2012-09-28"))
      
      # Save the predicted sales data as a CSV file
      write.csv(predicted_sales_data, file, row.names = FALSE)
    }
  )
  
  
  # Server - Correlation Map
  output$correlation_map <- renderPlot({
    df <- processed_data()
    if (!is.null(df)) {
      numeric_cols <- sapply(df, is.numeric)
      corr_matrix <- cor(df[, numeric_cols])
      corrplot(corr_matrix, method = "color")
    }
  })
}

shinyApp(ui, server)
