
pub const GWH_DELETER_HOST: &'static str = "gwh_deleter_host";
pub const GWH_DELETER_DEVICE: &'static str = "gwh_deleter_device";
pub const GWH_ALLOCATE_HOST: &'static str = "gwh_allocate_host";
pub const GWH_ALLOCATE_DEVICE: &'static str = "gwh_allocate_device";
pub const GWH_READ_AT: &'static str = "gwh_read_at";
pub const GWH_WRITE_AT: &'static str = "gwh_write_at";
pub const GWH_CHECK: &'static str = "gwh_check";

pub const GWH_THREADCOUNT: &'static str = "gwh_thread_count";
pub const GWH_BLOCKSIZE: &'static str = "gwh_blocksize";
pub const GWH_GRIDSIZE: &'static str = "gwh_gridsize";

pub const GWH_COPY_FOR_LOOP_INDEX_PREFIX: &'static str = "gwh_ci_" /* + dimensions */;
pub const GWH_KERNEL_PREFIX: &'static str = "gwh_kernel_" /* + unique number */;
pub const GWH_FOR_LOOP_INDEX_PREFIX: &'static str = "gwh_i_" /* + unique number */;
pub const GWH_FOR_LOOP_THREADCOUNT_PREFIX: &'static str = "gwh_thread_count_" /* + unique number */;

pub const GWH_ZERO_VIEW_STRUCT_NAME_PREFIX: &'static str = "GwhZeros" /* + dimensions */;