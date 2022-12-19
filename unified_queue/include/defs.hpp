#pragma once

// Cuda
#define BLOCK_SIZE 256

// Path
#define LINEITEM_PATH "data/lineitem.tbl"

// Query
#define TASK_SIZE 10000
#define DATE_BOTTOM_LIMIT 727841 
#define DATE_UPPER_LIMIT 728206 
#define QUANTITY_LIMIT 24.0 
#define DISCOUNT_BOTTOM_LIMIT 0.05
#define DISCOUNT_UPPER_LIMIT 0.07

// Task
#define CPU_TASK 0
#define GPU_TASK 1

// Mode
// #define MALLOCMANAGED
