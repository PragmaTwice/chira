; ModuleID = '/home/twice/projects/chira/chira/runtime/chirart.cpp'
source_filename = "/home/twice/projects/chira/chira/runtime/chirart.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"

%"struct.chirart::Var" = type { i64, %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { ptr, ptr }

@.str = private unnamed_addr constant [23 x i8] c"Argument size mismatch\00", align 1
@.str.1 = private unnamed_addr constant [19 x i8] c"Invalid tag in Var\00", align 1
@.str.2 = private unnamed_addr constant [26 x i8] c"Too many closure captures\00", align 1
@.str.3 = private unnamed_addr constant [42 x i8] c"Var is not a closure or primary operation\00", align 1
@stderr = external local_unnamed_addr global ptr, align 8
@.str.4 = private unnamed_addr constant [22 x i8] c"Assertion failed: %s\0A\00", align 1
@.str.5 = private unnamed_addr constant [21 x i8] c"Var is not a boolean\00", align 1
@.str.6 = private unnamed_addr constant [33 x i8] c"Invalid type to perform addition\00", align 1
@.str.9 = private unnamed_addr constant [17 x i8] c"Unreachable: %s\0A\00", align 1
@.str.10 = private unnamed_addr constant [36 x i8] c"Invalid type to perform subtraction\00", align 1
@.str.11 = private unnamed_addr constant [39 x i8] c"Invalid type to perform multiplication\00", align 1
@.str.12 = private unnamed_addr constant [17 x i8] c"Division by zero\00", align 1
@.str.13 = private unnamed_addr constant [33 x i8] c"Invalid type to perform division\00", align 1
@.str.14 = private unnamed_addr constant [35 x i8] c"Invalid type to perform comparison\00", align 1
@.str.15 = private unnamed_addr constant [39 x i8] c"Invalid type to perform equality check\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.16 = private unnamed_addr constant [4 x i8] c"%ld\00", align 1
@.str.17 = private unnamed_addr constant [4 x i8] c"%lf\00", align 1
@.str.18 = private unnamed_addr constant [20 x i8] c"Not implemented yet\00", align 1

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
  %8 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2) #11
  tail call void @abort() #12
  unreachable

9:                                                ; preds = %4
  %10 = or disjoint i64 %3, 131072
  store i64 %10, ptr %0, align 8, !tbaa !5
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %1, ptr %11, align 8, !tbaa !10
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %2, ptr %12, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_prim_op(ptr nocapture noundef writeonly %0, ptr noundef %1, i64 noundef %2) local_unnamed_addr #2 {
  %4 = icmp ult i64 %2, 65536
  br i1 %4, label %8, label %5, !prof !11

5:                                                ; preds = %3
  %6 = load ptr, ptr @stderr, align 8, !tbaa !12
  %7 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %6, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.2) #11
  tail call void @abort() #12
  unreachable

8:                                                ; preds = %3
  %9 = or disjoint i64 %2, 65536
  store i64 %9, ptr %0, align 8, !tbaa !5
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %1, ptr %10, align 8, !tbaa !10
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr null, ptr %11, align 8, !tbaa !10
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
  br label %44

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load double, ptr %9, align 8, !tbaa !10
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %10, ptr %11, align 8, !tbaa !10
  br label %44

12:                                               ; preds = %2
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i8, ptr %13, align 8, !tbaa !10, !range !15, !noundef !16
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %14, ptr %15, align 8, !tbaa !10
  br label %44

16:                                               ; preds = %2
  %17 = add i64 %3, -65536
  %18 = icmp ult i64 %17, 131072
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
  br label %44

26:                                               ; preds = %16
  switch i64 %3, label %41 [
    i64 4, label %27
    i64 7, label %27
    i64 5, label %34
    i64 6, label %44
    i64 0, label %44
  ]

27:                                               ; preds = %26, %26
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %29 = load ptr, ptr %28, align 8, !tbaa !10
  %30 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %29, ptr %30, align 8, !tbaa !10
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %32 = load i64, ptr %31, align 8, !tbaa !10
  %33 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %32, ptr %33, align 8, !tbaa !10
  br label %44

34:                                               ; preds = %26
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %36 = load ptr, ptr %35, align 8, !tbaa !10
  %37 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %36, ptr %37, align 8, !tbaa !10
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %39 = load ptr, ptr %38, align 8, !tbaa !10
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %39, ptr %40, align 8, !tbaa !10
  br label %44

41:                                               ; preds = %26
  %42 = load ptr, ptr @stderr, align 8, !tbaa !12
  %43 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %42, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1) #11
  tail call void @abort() #12
  unreachable

44:                                               ; preds = %4, %8, %12, %19, %26, %26, %27, %34
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_lambda(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  switch i64 %3, label %4 [
    i64 131072, label %7
    i64 65536, label %7
  ]

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !12
  %6 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %5, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3) #11
  tail call void @abort() #12
  unreachable

7:                                                ; preds = %1, %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !10
  ret ptr %9
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local ptr @chirart_get_env(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  switch i64 %3, label %4 [
    i64 131072, label %7
    i64 65536, label %7
  ]

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !12
  %6 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %5, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3) #11
  tail call void @abort() #12
  unreachable

7:                                                ; preds = %1, %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !10
  ret ptr %9
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

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_args_set_size(ptr nocapture noundef writeonly initializes((0, 8)) %0, i64 noundef %1) local_unnamed_addr #0 {
  store i64 %1, ptr %0, align 8, !tbaa !19
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @chirart_args_load(ptr noundef readnone %0, i64 noundef %1) local_unnamed_addr #4 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = getelementptr inbounds nuw [0 x %"struct.chirart::Var"], ptr %3, i64 0, i64 %1
  ret ptr %4
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_args_store(ptr nocapture noundef writeonly %0, i64 noundef %1, ptr nocapture noundef readonly %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = getelementptr inbounds nuw [0 x %"struct.chirart::Var"], ptr %4, i64 0, i64 %1
  %6 = load i64, ptr %2, align 8, !tbaa !5
  store i64 %6, ptr %5, align 8, !tbaa !5
  switch i64 %6, label %19 [
    i64 1, label %7
    i64 2, label %11
    i64 3, label %15
  ]

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %9 = load i64, ptr %8, align 8, !tbaa !10
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 %9, ptr %10, align 8, !tbaa !10
  br label %47

11:                                               ; preds = %3
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load double, ptr %12, align 8, !tbaa !10
  %14 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store double %13, ptr %14, align 8, !tbaa !10
  br label %47

15:                                               ; preds = %3
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %17 = load i8, ptr %16, align 8, !tbaa !10, !range !15, !noundef !16
  %18 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i8 %17, ptr %18, align 8, !tbaa !10
  br label %47

19:                                               ; preds = %3
  %20 = add i64 %6, -65536
  %21 = icmp ult i64 %20, 131072
  br i1 %21, label %22, label %29

22:                                               ; preds = %19
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %24 = load ptr, ptr %23, align 8, !tbaa !10
  %25 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %24, ptr %25, align 8, !tbaa !10
  %26 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %27 = load ptr, ptr %26, align 8, !tbaa !10
  %28 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %27, ptr %28, align 8, !tbaa !10
  br label %47

29:                                               ; preds = %19
  switch i64 %6, label %44 [
    i64 4, label %30
    i64 7, label %30
    i64 5, label %37
    i64 6, label %47
    i64 0, label %47
  ]

30:                                               ; preds = %29, %29
  %31 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %32 = load ptr, ptr %31, align 8, !tbaa !10
  %33 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %32, ptr %33, align 8, !tbaa !10
  %34 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %35 = load i64, ptr %34, align 8, !tbaa !10
  %36 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i64 %35, ptr %36, align 8, !tbaa !10
  br label %47

37:                                               ; preds = %29
  %38 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %39 = load ptr, ptr %38, align 8, !tbaa !10
  %40 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store ptr %39, ptr %40, align 8, !tbaa !10
  %41 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %42 = load ptr, ptr %41, align 8, !tbaa !10
  %43 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store ptr %42, ptr %43, align 8, !tbaa !10
  br label %47

44:                                               ; preds = %29
  %45 = load ptr, ptr @stderr, align 8, !tbaa !12
  %46 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %45, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1) #11
  tail call void @abort() #12
  unreachable

47:                                               ; preds = %7, %11, %15, %22, %29, %29, %30, %37
  ret void
}

; Function Attrs: mustprogress uwtable
define dso_local void @chirart_call(ptr noundef %0, ptr nocapture noundef readonly %1, ptr noundef %2) local_unnamed_addr #5 {
  %4 = load i64, ptr %2, align 8, !tbaa !19
  %5 = load i64, ptr %1, align 8, !tbaa !5
  %6 = and i64 %5, -65536
  switch i64 %6, label %7 [
    i64 131072, label %10
    i64 65536, label %13
  ], !prof !21

7:                                                ; preds = %3
  %8 = load ptr, ptr @stderr, align 8, !tbaa !12
  %9 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %8, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.3) #11
  tail call void @abort() #12
  unreachable

10:                                               ; preds = %3
  %11 = add nsw i64 %5, -131072
  %12 = icmp eq i64 %4, %11
  br i1 %12, label %19, label %16, !prof !11

13:                                               ; preds = %3
  %14 = add nsw i64 %5, -65536
  %15 = icmp eq i64 %4, %14
  br i1 %15, label %19, label %16, !prof !11

16:                                               ; preds = %13, %10
  %17 = load ptr, ptr @stderr, align 8, !tbaa !12
  %18 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %17, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str) #11
  tail call void @abort() #12
  unreachable

19:                                               ; preds = %13, %10
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %21 = load ptr, ptr %20, align 8, !tbaa !10
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %23 = load ptr, ptr %22, align 8, !tbaa !10
  tail call void %21(ptr noundef %0, ptr noundef nonnull %2, ptr noundef %23)
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local zeroext i1 @chirart_get_bool(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = icmp eq i64 %2, 3
  br i1 %3, label %7, label %4, !prof !11

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !12
  %6 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %5, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.5) #11
  tail call void @abort() #12
  unreachable

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = load i8, ptr %8, align 8, !tbaa !10, !range !15, !noundef !16
  %10 = trunc nuw i8 %9 to i1
  ret i1 %10
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_add(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !22
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %11

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !22
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %36, label %15

11:                                               ; preds = %3
  %12 = icmp eq i64 %6, 2
  br i1 %12, label %13, label %33

13:                                               ; preds = %11
  %14 = load i64, ptr %5, align 8, !tbaa !5, !noalias !22
  br label %15

15:                                               ; preds = %8, %13
  %16 = phi i64 [ %14, %13 ], [ %9, %8 ]
  %17 = add i64 %16, -1
  %18 = icmp ult i64 %17, 2
  br i1 %18, label %19, label %33

19:                                               ; preds = %15
  %20 = icmp eq i64 %16, 2
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %22 = load double, ptr %21, align 8, !noalias !22
  %23 = bitcast double %22 to i64
  %24 = sitofp i64 %23 to double
  %25 = select i1 %7, double %24, double %22
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %27 = load double, ptr %26, align 8, !noalias !22
  %28 = bitcast double %27 to i64
  %29 = sitofp i64 %28 to double
  %30 = select i1 %20, double %27, double %29
  %31 = fadd double %25, %30
  store i64 2, ptr %0, align 8, !tbaa !5
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %31, ptr %32, align 8, !tbaa !10
  br label %43

33:                                               ; preds = %15, %11
  %34 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !22
  %35 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %34, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.6) #11, !noalias !22
  tail call void @abort() #12, !noalias !22
  unreachable

36:                                               ; preds = %8
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %38 = load i64, ptr %37, align 8, !tbaa !10, !noalias !22
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %40 = load i64, ptr %39, align 8, !tbaa !10, !noalias !22
  %41 = add nsw i64 %40, %38
  store i64 1, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %41, ptr %42, align 8, !tbaa !10
  br label %43

43:                                               ; preds = %36, %19
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_sub(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !25
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %11

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !25
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %36, label %15

11:                                               ; preds = %3
  %12 = icmp eq i64 %6, 2
  br i1 %12, label %13, label %33

13:                                               ; preds = %11
  %14 = load i64, ptr %5, align 8, !tbaa !5, !noalias !25
  br label %15

15:                                               ; preds = %8, %13
  %16 = phi i64 [ %14, %13 ], [ %9, %8 ]
  %17 = add i64 %16, -1
  %18 = icmp ult i64 %17, 2
  br i1 %18, label %19, label %33

19:                                               ; preds = %15
  %20 = icmp eq i64 %16, 2
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %22 = load double, ptr %21, align 8, !noalias !25
  %23 = bitcast double %22 to i64
  %24 = sitofp i64 %23 to double
  %25 = select i1 %7, double %24, double %22
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %27 = load double, ptr %26, align 8, !noalias !25
  %28 = bitcast double %27 to i64
  %29 = sitofp i64 %28 to double
  %30 = select i1 %20, double %27, double %29
  %31 = fsub double %25, %30
  store i64 2, ptr %0, align 8, !tbaa !5
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %31, ptr %32, align 8, !tbaa !10
  br label %43

33:                                               ; preds = %15, %11
  %34 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !25
  %35 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %34, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.10) #11, !noalias !25
  tail call void @abort() #12, !noalias !25
  unreachable

36:                                               ; preds = %8
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %38 = load i64, ptr %37, align 8, !tbaa !10, !noalias !25
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %40 = load i64, ptr %39, align 8, !tbaa !10, !noalias !25
  %41 = sub nsw i64 %38, %40
  store i64 1, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %41, ptr %42, align 8, !tbaa !10
  br label %43

43:                                               ; preds = %36, %19
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_mul(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !28
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %11

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !28
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %36, label %15

11:                                               ; preds = %3
  %12 = icmp eq i64 %6, 2
  br i1 %12, label %13, label %33

13:                                               ; preds = %11
  %14 = load i64, ptr %5, align 8, !tbaa !5, !noalias !28
  br label %15

15:                                               ; preds = %8, %13
  %16 = phi i64 [ %14, %13 ], [ %9, %8 ]
  %17 = add i64 %16, -1
  %18 = icmp ult i64 %17, 2
  br i1 %18, label %19, label %33

19:                                               ; preds = %15
  %20 = icmp eq i64 %16, 2
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %22 = load double, ptr %21, align 8, !noalias !28
  %23 = bitcast double %22 to i64
  %24 = sitofp i64 %23 to double
  %25 = select i1 %7, double %24, double %22
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %27 = load double, ptr %26, align 8, !noalias !28
  %28 = bitcast double %27 to i64
  %29 = sitofp i64 %28 to double
  %30 = select i1 %20, double %27, double %29
  %31 = fmul double %25, %30
  store i64 2, ptr %0, align 8, !tbaa !5
  %32 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %31, ptr %32, align 8, !tbaa !10
  br label %43

33:                                               ; preds = %15, %11
  %34 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !28
  %35 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %34, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.11) #11, !noalias !28
  tail call void @abort() #12, !noalias !28
  unreachable

36:                                               ; preds = %8
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %38 = load i64, ptr %37, align 8, !tbaa !10, !noalias !28
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %40 = load i64, ptr %39, align 8, !tbaa !10, !noalias !28
  %41 = mul nsw i64 %40, %38
  store i64 1, ptr %0, align 8, !tbaa !5
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %41, ptr %42, align 8, !tbaa !10
  br label %43

43:                                               ; preds = %36, %19
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_div(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !31
  %6 = add i64 %5, -1
  %7 = icmp ult i64 %6, 2
  br i1 %7, label %8, label %24

8:                                                ; preds = %3
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %10 = load i64, ptr %9, align 8, !tbaa !5, !noalias !31
  %11 = add i64 %10, -1
  %12 = icmp ult i64 %11, 2
  br i1 %12, label %13, label %24

13:                                               ; preds = %8
  %14 = icmp eq i64 %10, 2
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %16 = load double, ptr %15, align 8, !noalias !31
  %17 = bitcast double %16 to i64
  %18 = sitofp i64 %17 to double
  %19 = select i1 %14, double %16, double %18
  %20 = fcmp une double %19, 0.000000e+00
  br i1 %20, label %27, label %21, !prof !11

21:                                               ; preds = %13
  %22 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !31
  %23 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %22, ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.12) #11, !noalias !31
  tail call void @abort() #12, !noalias !31
  unreachable

24:                                               ; preds = %8, %3
  %25 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !31
  %26 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %25, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.13) #11, !noalias !31
  tail call void @abort() #12, !noalias !31
  unreachable

27:                                               ; preds = %13
  %28 = icmp eq i64 %5, 2
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %30 = load double, ptr %29, align 8, !noalias !31
  %31 = bitcast double %30 to i64
  %32 = sitofp i64 %31 to double
  %33 = select i1 %28, double %30, double %32
  %34 = fdiv double %33, %19
  store i64 2, ptr %0, align 8, !tbaa !5
  %35 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %34, ptr %35, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_lt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !34
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !34
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !34
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !34
  %16 = icmp slt i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !34
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !34
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !34
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp olt double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !34
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.14) #11, !noalias !34
  tail call void @abort() #12, !noalias !34
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %37, %25 ], [ %16, %11 ]
  %43 = zext i1 %42 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_le(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !37
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !37
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !37
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !37
  %16 = icmp sle i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !37
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !37
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !37
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp ole double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !37
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.14) #11, !noalias !37
  tail call void @abort() #12, !noalias !37
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %37, %25 ], [ %16, %11 ]
  %43 = zext i1 %42 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_gt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !40
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !40
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !40
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !40
  %16 = icmp sgt i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !40
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !40
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !40
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp ogt double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !40
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.14) #11, !noalias !40
  tail call void @abort() #12, !noalias !40
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %37, %25 ], [ %16, %11 ]
  %43 = zext i1 %42 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_ge(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !43
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !43
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !43
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !43
  %16 = icmp sge i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !43
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !43
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !43
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp oge double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !43
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.14) #11, !noalias !43
  tail call void @abort() #12, !noalias !43
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %37, %25 ], [ %16, %11 ]
  %43 = zext i1 %42 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_eq(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !46
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !46
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !46
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !46
  %16 = icmp eq i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !46
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !46
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !46
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp oeq double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !46
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.15) #11, !noalias !46
  tail call void @abort() #12, !noalias !46
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %37, %25 ], [ %16, %11 ]
  %43 = zext i1 %42 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_display(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5
  switch i64 %5, label %16 [
    i64 1, label %6
    i64 2, label %11
  ]

6:                                                ; preds = %3
  %7 = load ptr, ptr @stdout, align 8, !tbaa !12
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %9 = load i64, ptr %8, align 8, !tbaa !10
  %10 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.16, i64 noundef %9) #13
  br label %19

11:                                               ; preds = %3
  %12 = load ptr, ptr @stdout, align 8, !tbaa !12
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %14 = load double, ptr %13, align 8
  %15 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %12, ptr noundef nonnull @.str.17, double noundef %14) #13
  br label %19

16:                                               ; preds = %3
  %17 = load ptr, ptr @stderr, align 8, !tbaa !12
  %18 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %17, ptr noundef nonnull @.str.9, ptr noundef nonnull @.str.18) #11
  tail call void @abort() #12
  unreachable

19:                                               ; preds = %6, %11
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @chirart_newline(ptr nocapture noundef writeonly initializes((0, 8)) %0, ptr nocapture noundef readnone %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = load ptr, ptr @stdout, align 8, !tbaa !12
  %5 = tail call i32 @fputc(i32 10, ptr %4)
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 {
  %1 = alloca %"struct.chirart::Var", align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1) #13
  store i64 0, ptr %1, align 8, !tbaa !5
  call void @chiracg_main(ptr noundef nonnull %1, ptr noundef null, ptr noundef null)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1) #13
  ret i32 0
}

declare void @chiracg_main(ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #8

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #9

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, ptr nocapture noundef) local_unnamed_addr #10

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { cold nofree noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { nofree nounwind }
attributes #11 = { cold nounwind }
attributes #12 = { noreturn nounwind }
attributes #13 = { nounwind }

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
!19 = !{!20, !20, i64 0}
!20 = !{!"long", !8, i64 0}
!21 = !{!"branch_weights", i32 2146410, i32 -2147483648, i32 2145337238}
!22 = !{!23}
!23 = distinct !{!23, !24, !"_ZN7chirartplERKNS_3VarES2_: argument 0"}
!24 = distinct !{!24, !"_ZN7chirartplERKNS_3VarES2_"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"_ZN7chirartmiERKNS_3VarES2_: argument 0"}
!27 = distinct !{!27, !"_ZN7chirartmiERKNS_3VarES2_"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"_ZN7chirartmlERKNS_3VarES2_: argument 0"}
!30 = distinct !{!30, !"_ZN7chirartmlERKNS_3VarES2_"}
!31 = !{!32}
!32 = distinct !{!32, !33, !"_ZN7chirartdvERKNS_3VarES2_: argument 0"}
!33 = distinct !{!33, !"_ZN7chirartdvERKNS_3VarES2_"}
!34 = !{!35}
!35 = distinct !{!35, !36, !"_ZN7chirartltERKNS_3VarES2_: argument 0"}
!36 = distinct !{!36, !"_ZN7chirartltERKNS_3VarES2_"}
!37 = !{!38}
!38 = distinct !{!38, !39, !"_ZN7chirartleERKNS_3VarES2_: argument 0"}
!39 = distinct !{!39, !"_ZN7chirartleERKNS_3VarES2_"}
!40 = !{!41}
!41 = distinct !{!41, !42, !"_ZN7chirartgtERKNS_3VarES2_: argument 0"}
!42 = distinct !{!42, !"_ZN7chirartgtERKNS_3VarES2_"}
!43 = !{!44}
!44 = distinct !{!44, !45, !"_ZN7chirartgeERKNS_3VarES2_: argument 0"}
!45 = distinct !{!45, !"_ZN7chirartgeERKNS_3VarES2_"}
!46 = !{!47}
!47 = distinct !{!47, !48, !"_ZN7chirarteqERKNS_3VarES2_: argument 0"}
!48 = distinct !{!48, !"_ZN7chirarteqERKNS_3VarES2_"}
