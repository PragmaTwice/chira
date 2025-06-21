; ModuleID = '/home/twice/projects/chira/chira/runtime/chirart.cpp'
source_filename = "/home/twice/projects/chira/chira/runtime/chirart.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"

%"struct.chirart::Var" = type { i64, %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { ptr, ptr }

@.str = private unnamed_addr constant [19 x i8] c"Invalid tag in Var\00", align 1
@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [22 x i8] c"Assertion failed: %s\0A\00", align 1
@.str.2 = private unnamed_addr constant [26 x i8] c"Too many closure captures\00", align 1
@.str.3 = private unnamed_addr constant [21 x i8] c"Var is not a closure\00", align 1
@.str.4 = private unnamed_addr constant [21 x i8] c"Var is not a boolean\00", align 1
@.str.5 = private unnamed_addr constant [33 x i8] c"Invalid type to perform addition\00", align 1
@.str.8 = private unnamed_addr constant [17 x i8] c"Unreachable: %s\0A\00", align 1
@.str.9 = private unnamed_addr constant [36 x i8] c"Invalid type to perform subtraction\00", align 1
@.str.10 = private unnamed_addr constant [39 x i8] c"Invalid type to perform multiplication\00", align 1
@.str.11 = private unnamed_addr constant [17 x i8] c"Division by zero\00", align 1
@.str.12 = private unnamed_addr constant [33 x i8] c"Invalid type to perform division\00", align 1
@.str.13 = private unnamed_addr constant [35 x i8] c"Invalid type to perform comparison\00", align 1
@.str.14 = private unnamed_addr constant [39 x i8] c"Invalid type to perform equality check\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.15 = private unnamed_addr constant [4 x i8] c"%ld\00", align 1
@.str.16 = private unnamed_addr constant [4 x i8] c"%lf\00", align 1
@.str.17 = private unnamed_addr constant [20 x i8] c"Not implemented yet\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_unspec(ptr nocapture noundef writeonly initializes((0, 8)) %0) local_unnamed_addr #0 {
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_int(ptr nocapture noundef writeonly initializes((0, 16)) %0, i64 noundef %1) local_unnamed_addr #0 {
  store i64 1, ptr %0, align 8, !tbaa !5
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %1, ptr %3, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_float(ptr nocapture noundef writeonly initializes((0, 16)) %0, double noundef %1) local_unnamed_addr #0 {
  store i64 2, ptr %0, align 8, !tbaa !5
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %1, ptr %3, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_closure(ptr nocapture noundef writeonly %0, ptr noundef %1, ptr noundef %2, i64 noundef %3) local_unnamed_addr #2 {
  %5 = icmp ult i64 %3, 65536
  br i1 %5, label %9, label %6, !prof !11

6:                                                ; preds = %4
  %7 = load ptr, ptr @stderr, align 8, !tbaa !12
  %8 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.2) #9
  tail call void @abort() #10
  unreachable

9:                                                ; preds = %4
  %10 = or disjoint i64 %3, 65536
  store i64 %10, ptr %0, align 8, !tbaa !5
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %1, ptr %11, align 8, !tbaa !10
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %2, ptr %12, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_set(ptr nocapture noundef writeonly initializes((0, 8)) %0, ptr nocapture noundef readonly %1) local_unnamed_addr #2 {
  %3 = load i64, ptr %1, align 8, !tbaa !5
  store i64 %3, ptr %0, align 8, !tbaa !5
  switch i64 %3, label %16 [
    i64 1, label %4
    i64 2, label %8
    i64 3, label %12
  ]

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load i64, ptr %5, align 8, !tbaa !10
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %6, ptr %7, align 8, !tbaa !10
  br label %31

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load double, ptr %9, align 8, !tbaa !10
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %10, ptr %11, align 8, !tbaa !10
  br label %31

12:                                               ; preds = %2
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i8, ptr %13, align 8, !tbaa !10, !range !15, !noundef !16
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %14, ptr %15, align 8, !tbaa !10
  br label %31

16:                                               ; preds = %2
  %17 = and i64 %3, -65536
  %18 = icmp eq i64 %17, 65536
  br i1 %18, label %19, label %26

19:                                               ; preds = %16
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %21 = load ptr, ptr %20, align 8, !tbaa !10
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %21, ptr %22, align 8, !tbaa !10
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %24 = load ptr, ptr %23, align 8, !tbaa !10
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %24, ptr %25, align 8, !tbaa !10
  br label %31

26:                                               ; preds = %16
  %27 = icmp eq i64 %3, 0
  br i1 %27, label %31, label %28

28:                                               ; preds = %26
  %29 = load ptr, ptr @stderr, align 8, !tbaa !12
  %30 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %29, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str) #9
  tail call void @abort() #10
  unreachable

31:                                               ; preds = %4, %8, %12, %19, %26
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_func_ptr(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %8, label %5, !prof !11

5:                                                ; preds = %1
  %6 = load ptr, ptr @stderr, align 8, !tbaa !12
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.3) #9
  tail call void @abort() #10
  unreachable

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !10
  ret ptr %10
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_caps(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %8, label %5, !prof !11

5:                                                ; preds = %1
  %6 = load ptr, ptr @stderr, align 8, !tbaa !12
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.3) #9
  tail call void @abort() #10
  unreachable

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %10 = load ptr, ptr %9, align 8, !tbaa !10
  ret ptr %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local ptr @chirart_env_load(ptr nocapture noundef readonly %0, i64 noundef %1) local_unnamed_addr #3 {
  %3 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  %4 = load ptr, ptr %3, align 8, !tbaa !17
  ret ptr %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_env_store(ptr nocapture noundef writeonly %0, i64 noundef %1, ptr noundef %2) local_unnamed_addr #0 {
  %4 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  store ptr %2, ptr %4, align 8, !tbaa !17
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local zeroext i1 @chirart_get_bool(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = icmp eq i64 %2, 3
  br i1 %3, label %7, label %4, !prof !11

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !12
  %6 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %5, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.4) #9
  tail call void @abort() #10
  unreachable

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load i8, ptr %8, align 8, !tbaa !10, !range !15, !noundef !16
  %10 = trunc nuw i8 %9 to i1
  ret i1 %10
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_add(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !19
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !19
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %34, label %13

9:                                                ; preds = %3
  %10 = icmp eq i64 %4, 2
  br i1 %10, label %11, label %31

11:                                               ; preds = %9
  %12 = load i64, ptr %2, align 8, !tbaa !5, !noalias !19
  br label %13

13:                                               ; preds = %6, %11
  %14 = phi i64 [ %12, %11 ], [ %7, %6 ]
  %15 = add i64 %14, -1
  %16 = icmp ult i64 %15, 2
  br i1 %16, label %17, label %31

17:                                               ; preds = %13
  %18 = icmp eq i64 %14, 2
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %20 = load double, ptr %19, align 8, !noalias !19
  %21 = bitcast double %20 to i64
  %22 = sitofp i64 %21 to double
  %23 = select i1 %5, double %22, double %20
  %24 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %25 = load double, ptr %24, align 8, !noalias !19
  %26 = bitcast double %25 to i64
  %27 = sitofp i64 %26 to double
  %28 = select i1 %18, double %25, double %27
  %29 = fadd double %23, %28
  store i64 2, ptr %0, align 8, !tbaa !5
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %29, ptr %30, align 8, !tbaa !10
  br label %41

31:                                               ; preds = %13, %9
  %32 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %33 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %32, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.5) #9, !noalias !19
  tail call void @abort() #10, !noalias !19
  unreachable

34:                                               ; preds = %6
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %36 = load i64, ptr %35, align 8, !tbaa !10, !noalias !19
  %37 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %38 = load i64, ptr %37, align 8, !tbaa !10, !noalias !19
  %39 = add nsw i64 %38, %36
  store i64 1, ptr %0, align 8, !tbaa !5
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %39, ptr %40, align 8, !tbaa !10
  br label %41

41:                                               ; preds = %34, %17
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_subtract(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !22
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !22
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %34, label %13

9:                                                ; preds = %3
  %10 = icmp eq i64 %4, 2
  br i1 %10, label %11, label %31

11:                                               ; preds = %9
  %12 = load i64, ptr %2, align 8, !tbaa !5, !noalias !22
  br label %13

13:                                               ; preds = %6, %11
  %14 = phi i64 [ %12, %11 ], [ %7, %6 ]
  %15 = add i64 %14, -1
  %16 = icmp ult i64 %15, 2
  br i1 %16, label %17, label %31

17:                                               ; preds = %13
  %18 = icmp eq i64 %14, 2
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %20 = load double, ptr %19, align 8, !noalias !22
  %21 = bitcast double %20 to i64
  %22 = sitofp i64 %21 to double
  %23 = select i1 %5, double %22, double %20
  %24 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %25 = load double, ptr %24, align 8, !noalias !22
  %26 = bitcast double %25 to i64
  %27 = sitofp i64 %26 to double
  %28 = select i1 %18, double %25, double %27
  %29 = fsub double %23, %28
  store i64 2, ptr %0, align 8, !tbaa !5
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %29, ptr %30, align 8, !tbaa !10
  br label %41

31:                                               ; preds = %13, %9
  %32 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !22
  %33 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %32, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.9) #9, !noalias !22
  tail call void @abort() #10, !noalias !22
  unreachable

34:                                               ; preds = %6
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %36 = load i64, ptr %35, align 8, !tbaa !10, !noalias !22
  %37 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %38 = load i64, ptr %37, align 8, !tbaa !10, !noalias !22
  %39 = sub nsw i64 %36, %38
  store i64 1, ptr %0, align 8, !tbaa !5
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %39, ptr %40, align 8, !tbaa !10
  br label %41

41:                                               ; preds = %34, %17
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_multiply(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !25
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !25
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %34, label %13

9:                                                ; preds = %3
  %10 = icmp eq i64 %4, 2
  br i1 %10, label %11, label %31

11:                                               ; preds = %9
  %12 = load i64, ptr %2, align 8, !tbaa !5, !noalias !25
  br label %13

13:                                               ; preds = %6, %11
  %14 = phi i64 [ %12, %11 ], [ %7, %6 ]
  %15 = add i64 %14, -1
  %16 = icmp ult i64 %15, 2
  br i1 %16, label %17, label %31

17:                                               ; preds = %13
  %18 = icmp eq i64 %14, 2
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %20 = load double, ptr %19, align 8, !noalias !25
  %21 = bitcast double %20 to i64
  %22 = sitofp i64 %21 to double
  %23 = select i1 %5, double %22, double %20
  %24 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %25 = load double, ptr %24, align 8, !noalias !25
  %26 = bitcast double %25 to i64
  %27 = sitofp i64 %26 to double
  %28 = select i1 %18, double %25, double %27
  %29 = fmul double %23, %28
  store i64 2, ptr %0, align 8, !tbaa !5
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %29, ptr %30, align 8, !tbaa !10
  br label %41

31:                                               ; preds = %13, %9
  %32 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !25
  %33 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %32, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.10) #9, !noalias !25
  tail call void @abort() #10, !noalias !25
  unreachable

34:                                               ; preds = %6
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %36 = load i64, ptr %35, align 8, !tbaa !10, !noalias !25
  %37 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %38 = load i64, ptr %37, align 8, !tbaa !10, !noalias !25
  %39 = mul nsw i64 %38, %36
  store i64 1, ptr %0, align 8, !tbaa !5
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %39, ptr %40, align 8, !tbaa !10
  br label %41

41:                                               ; preds = %34, %17
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_divide(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !28
  %5 = add i64 %4, -1
  %6 = icmp ult i64 %5, 2
  br i1 %6, label %7, label %11

7:                                                ; preds = %3
  %8 = load i64, ptr %2, align 8, !tbaa !5, !noalias !28
  %9 = add i64 %8, -1
  %10 = icmp ult i64 %9, 2
  br i1 %10, label %14, label %11

11:                                               ; preds = %7, %3
  %12 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !28
  %13 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %12, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.12) #9, !noalias !28
  tail call void @abort() #10, !noalias !28
  unreachable

14:                                               ; preds = %7
  %15 = icmp eq i64 %8, 2
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %17 = load double, ptr %16, align 8, !noalias !28
  %18 = bitcast double %17 to i64
  %19 = sitofp i64 %18 to double
  %20 = select i1 %15, double %17, double %19
  %21 = fcmp une double %20, 0.000000e+00
  br i1 %21, label %25, label %22, !prof !11

22:                                               ; preds = %14
  %23 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !28
  %24 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %23, ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.11) #9, !noalias !28
  tail call void @abort() #10, !noalias !28
  unreachable

25:                                               ; preds = %14
  %26 = icmp eq i64 %4, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %28 = load double, ptr %27, align 8, !noalias !28
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %26, double %28, double %30
  %32 = fdiv double %31, %20
  store i64 2, ptr %0, align 8, !tbaa !5
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %32, ptr %33, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_lt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !31
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %15

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !31
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %9, label %19

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !31
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !31
  %14 = icmp slt i64 %11, %13
  br label %39

15:                                               ; preds = %3
  %16 = icmp eq i64 %4, 2
  br i1 %16, label %17, label %36

17:                                               ; preds = %15
  %18 = load i64, ptr %2, align 8, !tbaa !5, !noalias !31
  br label %19

19:                                               ; preds = %6, %17
  %20 = phi i64 [ %18, %17 ], [ %7, %6 ]
  %21 = add i64 %20, -1
  %22 = icmp ult i64 %21, 2
  br i1 %22, label %23, label %36

23:                                               ; preds = %19
  %24 = icmp eq i64 %20, 2
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %26 = load double, ptr %25, align 8, !noalias !31
  %27 = bitcast double %26 to i64
  %28 = sitofp i64 %27 to double
  %29 = select i1 %5, double %28, double %26
  %30 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %31 = load double, ptr %30, align 8, !noalias !31
  %32 = bitcast double %31 to i64
  %33 = sitofp i64 %32 to double
  %34 = select i1 %24, double %31, double %33
  %35 = fcmp olt double %29, %34
  br label %39

36:                                               ; preds = %19, %15
  %37 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !31
  %38 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %37, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.13) #9, !noalias !31
  tail call void @abort() #10, !noalias !31
  unreachable

39:                                               ; preds = %23, %9
  %40 = phi i1 [ %35, %23 ], [ %14, %9 ]
  %41 = zext i1 %40 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %41, ptr %42, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_le(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !34
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %15

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !34
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %9, label %19

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !34
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !34
  %14 = icmp sle i64 %11, %13
  br label %39

15:                                               ; preds = %3
  %16 = icmp eq i64 %4, 2
  br i1 %16, label %17, label %36

17:                                               ; preds = %15
  %18 = load i64, ptr %2, align 8, !tbaa !5, !noalias !34
  br label %19

19:                                               ; preds = %6, %17
  %20 = phi i64 [ %18, %17 ], [ %7, %6 ]
  %21 = add i64 %20, -1
  %22 = icmp ult i64 %21, 2
  br i1 %22, label %23, label %36

23:                                               ; preds = %19
  %24 = icmp eq i64 %20, 2
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %26 = load double, ptr %25, align 8, !noalias !34
  %27 = bitcast double %26 to i64
  %28 = sitofp i64 %27 to double
  %29 = select i1 %5, double %28, double %26
  %30 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %31 = load double, ptr %30, align 8, !noalias !34
  %32 = bitcast double %31 to i64
  %33 = sitofp i64 %32 to double
  %34 = select i1 %24, double %31, double %33
  %35 = fcmp ole double %29, %34
  br label %39

36:                                               ; preds = %19, %15
  %37 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !34
  %38 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %37, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.13) #9, !noalias !34
  tail call void @abort() #10, !noalias !34
  unreachable

39:                                               ; preds = %23, %9
  %40 = phi i1 [ %35, %23 ], [ %14, %9 ]
  %41 = zext i1 %40 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %41, ptr %42, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_gt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !37
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %15

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !37
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %9, label %19

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !37
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !37
  %14 = icmp sgt i64 %11, %13
  br label %39

15:                                               ; preds = %3
  %16 = icmp eq i64 %4, 2
  br i1 %16, label %17, label %36

17:                                               ; preds = %15
  %18 = load i64, ptr %2, align 8, !tbaa !5, !noalias !37
  br label %19

19:                                               ; preds = %6, %17
  %20 = phi i64 [ %18, %17 ], [ %7, %6 ]
  %21 = add i64 %20, -1
  %22 = icmp ult i64 %21, 2
  br i1 %22, label %23, label %36

23:                                               ; preds = %19
  %24 = icmp eq i64 %20, 2
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %26 = load double, ptr %25, align 8, !noalias !37
  %27 = bitcast double %26 to i64
  %28 = sitofp i64 %27 to double
  %29 = select i1 %5, double %28, double %26
  %30 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %31 = load double, ptr %30, align 8, !noalias !37
  %32 = bitcast double %31 to i64
  %33 = sitofp i64 %32 to double
  %34 = select i1 %24, double %31, double %33
  %35 = fcmp ogt double %29, %34
  br label %39

36:                                               ; preds = %19, %15
  %37 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !37
  %38 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %37, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.13) #9, !noalias !37
  tail call void @abort() #10, !noalias !37
  unreachable

39:                                               ; preds = %23, %9
  %40 = phi i1 [ %35, %23 ], [ %14, %9 ]
  %41 = zext i1 %40 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %41, ptr %42, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_ge(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !40
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %15

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !40
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %9, label %19

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !40
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !40
  %14 = icmp sge i64 %11, %13
  br label %39

15:                                               ; preds = %3
  %16 = icmp eq i64 %4, 2
  br i1 %16, label %17, label %36

17:                                               ; preds = %15
  %18 = load i64, ptr %2, align 8, !tbaa !5, !noalias !40
  br label %19

19:                                               ; preds = %6, %17
  %20 = phi i64 [ %18, %17 ], [ %7, %6 ]
  %21 = add i64 %20, -1
  %22 = icmp ult i64 %21, 2
  br i1 %22, label %23, label %36

23:                                               ; preds = %19
  %24 = icmp eq i64 %20, 2
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %26 = load double, ptr %25, align 8, !noalias !40
  %27 = bitcast double %26 to i64
  %28 = sitofp i64 %27 to double
  %29 = select i1 %5, double %28, double %26
  %30 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %31 = load double, ptr %30, align 8, !noalias !40
  %32 = bitcast double %31 to i64
  %33 = sitofp i64 %32 to double
  %34 = select i1 %24, double %31, double %33
  %35 = fcmp oge double %29, %34
  br label %39

36:                                               ; preds = %19, %15
  %37 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !40
  %38 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %37, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.13) #9, !noalias !40
  tail call void @abort() #10, !noalias !40
  unreachable

39:                                               ; preds = %23, %9
  %40 = phi i1 [ %35, %23 ], [ %14, %9 ]
  %41 = zext i1 %40 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %41, ptr %42, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_eq(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = load i64, ptr %1, align 8, !tbaa !5, !noalias !43
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %6, label %15

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8, !tbaa !5, !noalias !43
  %8 = icmp eq i64 %7, 1
  br i1 %8, label %9, label %19

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !43
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !43
  %14 = icmp eq i64 %11, %13
  br label %39

15:                                               ; preds = %3
  %16 = icmp eq i64 %4, 2
  br i1 %16, label %17, label %36

17:                                               ; preds = %15
  %18 = load i64, ptr %2, align 8, !tbaa !5, !noalias !43
  br label %19

19:                                               ; preds = %6, %17
  %20 = phi i64 [ %18, %17 ], [ %7, %6 ]
  %21 = add i64 %20, -1
  %22 = icmp ult i64 %21, 2
  br i1 %22, label %23, label %36

23:                                               ; preds = %19
  %24 = icmp eq i64 %20, 2
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %26 = load double, ptr %25, align 8, !noalias !43
  %27 = bitcast double %26 to i64
  %28 = sitofp i64 %27 to double
  %29 = select i1 %5, double %28, double %26
  %30 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %31 = load double, ptr %30, align 8, !noalias !43
  %32 = bitcast double %31 to i64
  %33 = sitofp i64 %32 to double
  %34 = select i1 %24, double %31, double %33
  %35 = fcmp oeq double %29, %34
  br label %39

36:                                               ; preds = %19, %15
  %37 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !43
  %38 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %37, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.14) #9, !noalias !43
  tail call void @abort() #10, !noalias !43
  unreachable

39:                                               ; preds = %23, %9
  %40 = phi i1 [ %35, %23 ], [ %14, %9 ]
  %41 = zext i1 %40 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %41, ptr %42, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_display(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #2 {
  %3 = load i64, ptr %1, align 8, !tbaa !5
  switch i64 %3, label %14 [
    i64 1, label %4
    i64 2, label %9
  ]

4:                                                ; preds = %2
  %5 = load ptr, ptr @stdout, align 8, !tbaa !12
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load i64, ptr %6, align 8, !tbaa !10
  %8 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %5, ptr noundef nonnull @.str.15, i64 noundef %7) #11
  br label %17

9:                                                ; preds = %2
  %10 = load ptr, ptr @stdout, align 8, !tbaa !12
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load double, ptr %11, align 8
  %13 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %10, ptr noundef nonnull @.str.16, double noundef %12) #11
  br label %17

14:                                               ; preds = %2
  %15 = load ptr, ptr @stderr, align 8, !tbaa !12
  %16 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %15, ptr noundef nonnull @.str.8, ptr noundef nonnull @.str.17) #9
  tail call void @abort() #10
  unreachable

17:                                               ; preds = %4, %9
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_newline(ptr nocapture noundef writeonly initializes((0, 8)) %0) local_unnamed_addr #2 {
  %2 = load ptr, ptr @stdout, align 8, !tbaa !12
  %3 = tail call i32 @fputc(i32 10, ptr %2)
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  %1 = alloca %"struct.chirart::Var", align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1) #11
  store i64 0, ptr %1, align 8, !tbaa !5
  call void @chiracg_main(ptr noundef nonnull %1, ptr noundef null)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1) #11
  ret i32 0
}

declare void @chiracg_main(ptr noundef, ptr noundef) local_unnamed_addr #5

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #6

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, ptr nocapture noundef) local_unnamed_addr #8

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { cold nofree noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { nofree nounwind }
attributes #9 = { cold nounwind }
attributes #10 = { noreturn nounwind }
attributes #11 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 20.1.3 (https://github.com/conda-forge/clangdev-feedstock 3e9dfa811865fe27bcd95c0004d27603f2ec4a73)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"_ZTSN7chirart3VarE", !7, i64 0, !8, i64 8}
!7 = !{!"_ZTSN7chirart3Var3TagE", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!12 = !{!13, !13, i64 0}
!13 = !{!"p1 _ZTS8_IO_FILE", !14, i64 0}
!14 = !{!"any pointer", !8, i64 0}
!15 = !{i8 0, i8 2}
!16 = !{}
!17 = !{!18, !18, i64 0}
!18 = !{!"p1 _ZTSN7chirart3VarE", !14, i64 0}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZN7chirartplERKNS_3VarES2_: argument 0"}
!21 = distinct !{!21, !"_ZN7chirartplERKNS_3VarES2_"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"_ZN7chirartmiERKNS_3VarES2_: argument 0"}
!24 = distinct !{!24, !"_ZN7chirartmiERKNS_3VarES2_"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"_ZN7chirartmlERKNS_3VarES2_: argument 0"}
!27 = distinct !{!27, !"_ZN7chirartmlERKNS_3VarES2_"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"_ZN7chirartdvERKNS_3VarES2_: argument 0"}
!30 = distinct !{!30, !"_ZN7chirartdvERKNS_3VarES2_"}
!31 = !{!32}
!32 = distinct !{!32, !33, !"_ZN7chirartltERKNS_3VarES2_: argument 0"}
!33 = distinct !{!33, !"_ZN7chirartltERKNS_3VarES2_"}
!34 = !{!35}
!35 = distinct !{!35, !36, !"_ZN7chirartleERKNS_3VarES2_: argument 0"}
!36 = distinct !{!36, !"_ZN7chirartleERKNS_3VarES2_"}
!37 = !{!38}
!38 = distinct !{!38, !39, !"_ZN7chirartgtERKNS_3VarES2_: argument 0"}
!39 = distinct !{!39, !"_ZN7chirartgtERKNS_3VarES2_"}
!40 = !{!41}
!41 = distinct !{!41, !42, !"_ZN7chirartgeERKNS_3VarES2_: argument 0"}
!42 = distinct !{!42, !"_ZN7chirartgeERKNS_3VarES2_"}
!43 = !{!44}
!44 = distinct !{!44, !45, !"_ZN7chirarteqERKNS_3VarES2_: argument 0"}
!45 = distinct !{!45, !"_ZN7chirarteqERKNS_3VarES2_"}
