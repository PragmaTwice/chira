; ModuleID = '/home/twice/projects/chira/chira/runtime/chirart.cpp'
source_filename = "/home/twice/projects/chira/chira/runtime/chirart.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"

%"struct.chirart::Var" = type { i64, %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { ptr, ptr }

@.str = private unnamed_addr constant [26 x i8] c"Too many closure captures\00", align 1
@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [19 x i8] c"Assertion failed: \00", align 1
@.str.3 = private unnamed_addr constant [47 x i8] c"Argument size mismatch (expected %zu, got %zu)\00", align 1
@.str.4 = private unnamed_addr constant [60 x i8] c"Argument size mismatch (expected no less than %zu, got %zu)\00", align 1
@.str.5 = private unnamed_addr constant [42 x i8] c"Var is not a closure or primary operation\00", align 1
@.str.6 = private unnamed_addr constant [21 x i8] c"Var is not a boolean\00", align 1
@.str.7 = private unnamed_addr constant [33 x i8] c"Invalid type to perform addition\00", align 1
@.str.10 = private unnamed_addr constant [17 x i8] c"Unreachable: %s\0A\00", align 1
@.str.11 = private unnamed_addr constant [36 x i8] c"Invalid type to perform subtraction\00", align 1
@.str.12 = private unnamed_addr constant [39 x i8] c"Invalid type to perform multiplication\00", align 1
@.str.13 = private unnamed_addr constant [17 x i8] c"Division by zero\00", align 1
@.str.14 = private unnamed_addr constant [33 x i8] c"Invalid type to perform division\00", align 1
@.str.15 = private unnamed_addr constant [35 x i8] c"Invalid type to perform comparison\00", align 1
@.str.16 = private unnamed_addr constant [39 x i8] c"Invalid type to perform equality check\00", align 1
@.str.17 = private unnamed_addr constant [41 x i8] c"Invalid type to perform logical negation\00", align 1
@.str.18 = private unnamed_addr constant [36 x i8] c"Invalid type to perform logical AND\00", align 1
@.str.19 = private unnamed_addr constant [35 x i8] c"Invalid type to perform logical OR\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.20 = private unnamed_addr constant [4 x i8] c"%ld\00", align 1
@.str.21 = private unnamed_addr constant [4 x i8] c"%lf\00", align 1
@.str.22 = private unnamed_addr constant [3 x i8] c"#t\00", align 1
@.str.23 = private unnamed_addr constant [3 x i8] c"#f\00", align 1
@.str.24 = private unnamed_addr constant [20 x i8] c"Not implemented yet\00", align 1

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_unspec(ptr nocapture noundef writeonly initializes((0, 8)) %0) local_unnamed_addr #0 {
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_int(ptr nocapture noundef writeonly initializes((0, 16)) %0, i64 noundef %1) local_unnamed_addr #0 {
  store i64 1, ptr %0, align 8, !tbaa !5
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %1, ptr %3, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_float(ptr nocapture noundef writeonly initializes((0, 16)) %0, double noundef %1) local_unnamed_addr #0 {
  store i64 2, ptr %0, align 8, !tbaa !5
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %1, ptr %3, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_closure(ptr nocapture noundef writeonly %0, ptr noundef %1, ptr noundef %2, i64 noundef %3) local_unnamed_addr #2 {
  %5 = icmp ult i64 %3, 65536
  br i1 %5, label %13, label %6, !prof !11

6:                                                ; preds = %4
  %7 = load ptr, ptr @stderr, align 8, !tbaa !12
  %8 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %7) #13
  %9 = load ptr, ptr @stderr, align 8, !tbaa !12
  %10 = tail call i64 @fwrite(ptr nonnull @.str, i64 25, i64 1, ptr %9) #13
  %11 = load ptr, ptr @stderr, align 8, !tbaa !12
  %12 = tail call i32 @fputc(i32 10, ptr %11)
  tail call void @abort() #14
  unreachable

13:                                               ; preds = %4
  %14 = or disjoint i64 %3, 131072
  store i64 %14, ptr %0, align 8, !tbaa !5
  %15 = ptrtoint ptr %1 to i64
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %15, ptr %16, align 8, !tbaa !10
  %17 = ptrtoint ptr %2 to i64
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %17, ptr %18, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_prim_op(ptr nocapture noundef writeonly %0, ptr noundef %1, i64 noundef %2) local_unnamed_addr #2 {
  %4 = icmp ult i64 %2, 65536
  br i1 %4, label %12, label %5, !prof !11

5:                                                ; preds = %3
  %6 = load ptr, ptr @stderr, align 8, !tbaa !12
  %7 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %6) #13
  %8 = load ptr, ptr @stderr, align 8, !tbaa !12
  %9 = tail call i64 @fwrite(ptr nonnull @.str, i64 25, i64 1, ptr %8) #13
  %10 = load ptr, ptr @stderr, align 8, !tbaa !12
  %11 = tail call i32 @fputc(i32 10, ptr %10)
  tail call void @abort() #14
  unreachable

12:                                               ; preds = %3
  %13 = or disjoint i64 %2, 65536
  store i64 %13, ptr %0, align 8, !tbaa !5
  %14 = ptrtoint ptr %1 to i64
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %14, ptr %15, align 8, !tbaa !10
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 0, ptr %16, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @chirart_set(ptr nocapture noundef writeonly initializes((0, 24)) %0, ptr nocapture noundef readonly %1) local_unnamed_addr #3 {
  %3 = load i64, ptr %1, align 8, !tbaa !5
  store i64 %3, ptr %0, align 8, !tbaa !5
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !10
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %5, ptr %6, align 8, !tbaa !10
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %8 = load i64, ptr %7, align 8, !tbaa !10
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %8, ptr %9, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local ptr @chirart_env_load(ptr nocapture noundef readonly %0, i64 noundef %1) local_unnamed_addr #4 {
  %3 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  %4 = load ptr, ptr %3, align 8, !tbaa !15
  ret ptr %4
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_env_store(ptr nocapture noundef writeonly %0, i64 noundef %1, ptr noundef %2) local_unnamed_addr #0 {
  %4 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  store ptr %2, ptr %4, align 8, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_args_set_size(ptr nocapture noundef writeonly initializes((0, 8)) %0, i64 noundef %1) local_unnamed_addr #0 {
  store i64 %1, ptr %0, align 8, !tbaa !17
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local nonnull ptr @chirart_args_load(ptr noundef readnone %0, i64 noundef %1) local_unnamed_addr #5 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = getelementptr inbounds nuw [0 x %"struct.chirart::Var"], ptr %3, i64 0, i64 %1
  ret ptr %4
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @chirart_args_store(ptr nocapture noundef writeonly %0, i64 noundef %1, ptr nocapture noundef readonly %2) local_unnamed_addr #3 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = getelementptr inbounds nuw [0 x %"struct.chirart::Var"], ptr %4, i64 0, i64 %1
  %6 = load i64, ptr %2, align 8, !tbaa !5
  store i64 %6, ptr %5, align 8, !tbaa !5
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %8 = load i64, ptr %7, align 8, !tbaa !10
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i64 %8, ptr %9, align 8, !tbaa !10
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %11 = load i64, ptr %10, align 8, !tbaa !10
  %12 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i64 %11, ptr %12, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress uwtable
define dso_local void @chirart_call(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr noundef %2) local_unnamed_addr #6 {
  %4 = alloca %"struct.chirart::Var", align 16
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4) #15
  tail call void @llvm.experimental.noalias.scope.decl(metadata !19)
  %5 = load i64, ptr %1, align 8, !tbaa !5, !noalias !19
  %6 = and i64 %5, -65536
  switch i64 %6, label %7 [
    i64 131072, label %14
    i64 65536, label %14
  ]

7:                                                ; preds = %3
  %8 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %9 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %8) #13, !noalias !19
  %10 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %11 = tail call i64 @fwrite(ptr nonnull @.str.5, i64 41, i64 1, ptr %10) #13
  %12 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %13 = tail call i32 @fputc(i32 10, ptr %12), !noalias !19
  tail call void @abort() #14, !noalias !19
  unreachable

14:                                               ; preds = %3, %3
  %15 = and i64 %5, 32767
  %16 = and i64 %5, 32768
  %17 = icmp eq i64 %16, 0
  %18 = load i64, ptr %2, align 8, !tbaa !17, !noalias !19
  br i1 %17, label %19, label %29

19:                                               ; preds = %14
  %20 = icmp eq i64 %18, %15
  br i1 %20, label %39, label %21, !prof !11

21:                                               ; preds = %19
  %22 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %23 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %22) #13, !noalias !19
  %24 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %25 = load i64, ptr %2, align 8, !tbaa !17, !noalias !19
  %26 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %24, ptr noundef nonnull @.str.3, i64 noundef %15, i64 noundef %25) #16, !noalias !19
  %27 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %28 = tail call i32 @fputc(i32 10, ptr %27), !noalias !19
  tail call void @abort() #14, !noalias !19
  unreachable

29:                                               ; preds = %14
  %30 = icmp ult i64 %18, %15
  br i1 %30, label %31, label %39, !prof !22

31:                                               ; preds = %29
  %32 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %33 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %32) #13, !noalias !19
  %34 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %35 = load i64, ptr %2, align 8, !tbaa !17, !noalias !19
  %36 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %34, ptr noundef nonnull @.str.4, i64 noundef %15, i64 noundef %35) #16, !noalias !19
  %37 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %38 = tail call i32 @fputc(i32 10, ptr %37), !noalias !19
  tail call void @abort() #14, !noalias !19
  unreachable

39:                                               ; preds = %29, %19
  store i64 0, ptr %4, align 16, !tbaa !5, !alias.scope !19
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %41 = load ptr, ptr %40, align 8, !tbaa !10, !noalias !19
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %43 = load ptr, ptr %42, align 8, !tbaa !10, !noalias !19
  call void %41(ptr noundef nonnull align 8 %4, ptr noundef nonnull %2, ptr noundef %43)
  %44 = load <2 x i64>, ptr %4, align 16, !tbaa !10
  store <2 x i64> %44, ptr %0, align 8, !tbaa !10
  %45 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %46 = load i64, ptr %45, align 16, !tbaa !10
  %47 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %46, ptr %47, align 8, !tbaa !10
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4) #15
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local zeroext i1 @chirart_get_bool(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = icmp eq i64 %2, 3
  br i1 %3, label %11, label %4, !prof !11

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !12
  %6 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %5) #13
  %7 = load ptr, ptr @stderr, align 8, !tbaa !12
  %8 = tail call i64 @fwrite(ptr nonnull @.str.6, i64 20, i64 1, ptr %7) #13
  %9 = load ptr, ptr @stderr, align 8, !tbaa !12
  %10 = tail call i32 @fputc(i32 10, ptr %9)
  tail call void @abort() #14
  unreachable

11:                                               ; preds = %1
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %13 = load i8, ptr %12, align 8, !tbaa !10, !range !23, !noundef !24
  %14 = trunc nuw i8 %13 to i1
  ret i1 %14
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_add(ptr nocapture noundef writeonly initializes((0, 16)) %0, ptr noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  store i64 1, ptr %0, align 8, !tbaa !5
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 0, ptr %4, align 8, !tbaa !10
  %5 = load i64, ptr %1, align 8, !tbaa !17
  %6 = mul nuw nsw i64 %5, 24
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 %6
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %9 = icmp eq i64 %5, 0
  br i1 %9, label %12, label %10

10:                                               ; preds = %3
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  br label %13

12:                                               ; preds = %44, %3
  ret void

13:                                               ; preds = %10, %44
  %14 = phi i64 [ 0, %10 ], [ %45, %44 ]
  %15 = phi double [ 0.000000e+00, %10 ], [ %49, %44 ]
  %16 = phi i64 [ 1, %10 ], [ %46, %44 ]
  %17 = phi ptr [ %11, %10 ], [ %47, %44 ]
  %18 = icmp eq i64 %16, 1
  %19 = load i64, ptr %17, align 8, !tbaa !5, !noalias !25
  %20 = icmp eq i64 %19, 1
  %21 = select i1 %18, i1 %20, i1 false
  br i1 %21, label %22, label %26

22:                                               ; preds = %13
  %23 = getelementptr inbounds nuw i8, ptr %17, i64 8
  %24 = load i64, ptr %23, align 8, !tbaa !10, !noalias !25
  %25 = add nsw i64 %24, %14
  br label %44

26:                                               ; preds = %13
  %27 = add i64 %19, -1
  %28 = icmp ult i64 %27, 2
  br i1 %28, label %29, label %41

29:                                               ; preds = %26
  %30 = icmp eq i64 %19, 2
  %31 = bitcast double %15 to i64
  %32 = sitofp i64 %31 to double
  %33 = select i1 %18, double %32, double %15
  %34 = getelementptr inbounds nuw i8, ptr %17, i64 8
  %35 = load double, ptr %34, align 8, !noalias !25
  %36 = bitcast double %35 to i64
  %37 = sitofp i64 %36 to double
  %38 = select i1 %30, double %35, double %37
  %39 = fadd double %33, %38
  %40 = bitcast double %39 to i64
  br label %44

41:                                               ; preds = %26
  %42 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !25
  %43 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %42, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.7) #16, !noalias !25
  tail call void @abort() #14, !noalias !25
  unreachable

44:                                               ; preds = %22, %29
  %45 = phi i64 [ %25, %22 ], [ %40, %29 ]
  %46 = phi i64 [ 1, %22 ], [ 2, %29 ]
  store i64 %46, ptr %0, align 8, !tbaa !5
  store i64 %45, ptr %4, align 8, !tbaa !10
  %47 = getelementptr inbounds nuw i8, ptr %17, i64 24
  %48 = icmp eq ptr %47, %8
  %49 = bitcast i64 %45 to double
  br i1 %48, label %12, label %13
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_sub(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !28
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !28
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !28
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !28
  %16 = sub nsw i64 %13, %15
  br label %42

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %39

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !28
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %39

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !28
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !28
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fsub double %31, %36
  %38 = bitcast double %37 to i64
  br label %42

39:                                               ; preds = %21, %17
  %40 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !28
  %41 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %40, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.11) #16, !noalias !28
  tail call void @abort() #14, !noalias !28
  unreachable

42:                                               ; preds = %11, %25
  %43 = phi i64 [ %16, %11 ], [ %38, %25 ]
  %44 = phi i64 [ 1, %11 ], [ 2, %25 ]
  store i64 %44, ptr %0, align 8, !tbaa !5
  %45 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %43, ptr %45, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_mul(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !31
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !31
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !31
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !31
  %16 = mul nsw i64 %15, %13
  br label %42

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %39

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !31
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %39

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !31
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !31
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fmul double %31, %36
  %38 = bitcast double %37 to i64
  br label %42

39:                                               ; preds = %21, %17
  %40 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !31
  %41 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %40, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.12) #16, !noalias !31
  tail call void @abort() #14, !noalias !31
  unreachable

42:                                               ; preds = %11, %25
  %43 = phi i64 [ %16, %11 ], [ %38, %25 ]
  %44 = phi i64 [ 1, %11 ], [ 2, %25 ]
  store i64 %44, ptr %0, align 8, !tbaa !5
  %45 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %43, ptr %45, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_div(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !34
  %6 = add i64 %5, -1
  %7 = icmp ult i64 %6, 2
  br i1 %7, label %8, label %13

8:                                                ; preds = %3
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %10 = load i64, ptr %9, align 8, !tbaa !5, !noalias !34
  %11 = add i64 %10, -1
  %12 = icmp ult i64 %11, 2
  br i1 %12, label %16, label %13

13:                                               ; preds = %8, %3
  %14 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !34
  %15 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %14, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.14) #16, !noalias !34
  tail call void @abort() #14, !noalias !34
  unreachable

16:                                               ; preds = %8
  %17 = icmp eq i64 %10, 2
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %19 = load double, ptr %18, align 8, !noalias !34
  %20 = bitcast double %19 to i64
  %21 = sitofp i64 %20 to double
  %22 = select i1 %17, double %19, double %21
  %23 = fcmp une double %22, 0.000000e+00
  br i1 %23, label %31, label %24, !prof !11

24:                                               ; preds = %16
  %25 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !34
  %26 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %25) #13, !noalias !34
  %27 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !34
  %28 = tail call i64 @fwrite(ptr nonnull @.str.13, i64 16, i64 1, ptr %27) #13
  %29 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !34
  %30 = tail call i32 @fputc(i32 10, ptr %29), !noalias !34
  tail call void @abort() #14, !noalias !34
  unreachable

31:                                               ; preds = %16
  %32 = icmp eq i64 %5, 2
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %34 = load double, ptr %33, align 8, !noalias !34
  %35 = bitcast double %34 to i64
  %36 = sitofp i64 %35 to double
  %37 = select i1 %32, double %34, double %36
  %38 = fdiv double %37, %22
  store i64 2, ptr %0, align 8, !tbaa !5
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %38, ptr %39, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_lt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
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
  %16 = icmp slt i64 %13, %15
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
  %37 = fcmp olt double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !37
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.15) #16, !noalias !37
  tail call void @abort() #14, !noalias !37
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %16, %11 ], [ %37, %25 ]
  %43 = zext i1 %42 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_le(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
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
  %16 = icmp sle i64 %13, %15
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
  %37 = fcmp ole double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !40
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.15) #16, !noalias !40
  tail call void @abort() #14, !noalias !40
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %16, %11 ], [ %37, %25 ]
  %43 = zext i1 %42 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_gt(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
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
  %16 = icmp sgt i64 %13, %15
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
  %37 = fcmp ogt double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !43
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.15) #16, !noalias !43
  tail call void @abort() #14, !noalias !43
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %16, %11 ], [ %37, %25 ]
  %43 = zext i1 %42 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_ge(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
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
  %16 = icmp sge i64 %13, %15
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
  %37 = fcmp oge double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !46
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.15) #16, !noalias !46
  tail call void @abort() #14, !noalias !46
  unreachable

41:                                               ; preds = %11, %25
  %42 = phi i1 [ %16, %11 ], [ %37, %25 ]
  %43 = zext i1 %42 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %44 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %43, ptr %44, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_eq(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !49
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !49
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %20

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !49
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !49
  %16 = icmp eq i64 %13, %15
  br label %49

17:                                               ; preds = %3
  switch i64 %6, label %46 [
    i64 2, label %18
    i64 3, label %37
  ]

18:                                               ; preds = %17
  %19 = load i64, ptr %5, align 8, !tbaa !5, !noalias !49
  br label %20

20:                                               ; preds = %8, %18
  %21 = phi i64 [ %19, %18 ], [ %9, %8 ]
  %22 = add i64 %21, -1
  %23 = icmp ult i64 %22, 2
  br i1 %23, label %24, label %46

24:                                               ; preds = %20
  %25 = icmp eq i64 %21, 2
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %27 = load double, ptr %26, align 8, !noalias !49
  %28 = bitcast double %27 to i64
  %29 = sitofp i64 %28 to double
  %30 = select i1 %7, double %29, double %27
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %32 = load double, ptr %31, align 8, !noalias !49
  %33 = bitcast double %32 to i64
  %34 = sitofp i64 %33 to double
  %35 = select i1 %25, double %32, double %34
  %36 = fcmp oeq double %30, %35
  br label %49

37:                                               ; preds = %17
  %38 = load i64, ptr %5, align 8, !tbaa !5, !noalias !49
  %39 = icmp eq i64 %38, 3
  br i1 %39, label %40, label %46

40:                                               ; preds = %37
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %42 = load i8, ptr %41, align 8, !tbaa !10, !range !23, !noalias !49, !noundef !24
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %44 = load i8, ptr %43, align 8, !tbaa !10, !range !23, !noalias !49, !noundef !24
  %45 = icmp eq i8 %42, %44
  br label %49

46:                                               ; preds = %20, %17, %37
  %47 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !49
  %48 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %47, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.16) #16, !noalias !49
  tail call void @abort() #14, !noalias !49
  unreachable

49:                                               ; preds = %11, %24, %40
  %50 = phi i1 [ %16, %11 ], [ %36, %24 ], [ %45, %40 ]
  %51 = zext i1 %50 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %51, ptr %52, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_not(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !52
  %6 = icmp eq i64 %5, 3
  br i1 %6, label %10, label %7

7:                                                ; preds = %3
  %8 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !52
  %9 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %8, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.17) #16, !noalias !52
  tail call void @abort() #14, !noalias !52
  unreachable

10:                                               ; preds = %3
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %12 = load i8, ptr %11, align 8, !tbaa !10, !range !23, !noalias !52, !noundef !24
  %13 = xor i8 %12, 1
  %14 = zext nneg i8 %13 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %14, ptr %15, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_and(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !55
  %6 = icmp eq i64 %5, 3
  br i1 %6, label %7, label %19

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %9 = load i64, ptr %8, align 8, !tbaa !5, !noalias !55
  %10 = icmp eq i64 %9, 3
  br i1 %10, label %11, label %19

11:                                               ; preds = %7
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i8, ptr %12, align 8, !tbaa !10, !range !23, !noalias !55, !noundef !24
  %14 = trunc nuw i8 %13 to i1
  br i1 %14, label %15, label %22

15:                                               ; preds = %11
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %17 = load i8, ptr %16, align 8, !tbaa !10, !range !23, !noalias !55, !noundef !24
  %18 = zext nneg i8 %17 to i64
  br label %22

19:                                               ; preds = %7, %3
  %20 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !55
  %21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %20, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.18) #16, !noalias !55
  tail call void @abort() #14, !noalias !55
  unreachable

22:                                               ; preds = %11, %15
  %23 = phi i64 [ 0, %11 ], [ %18, %15 ]
  store i64 3, ptr %0, align 8, !tbaa !5
  %24 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %23, ptr %24, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_or(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !58
  %6 = icmp eq i64 %5, 3
  br i1 %6, label %7, label %19

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %9 = load i64, ptr %8, align 8, !tbaa !5, !noalias !58
  %10 = icmp eq i64 %9, 3
  br i1 %10, label %11, label %19

11:                                               ; preds = %7
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i8, ptr %12, align 8, !tbaa !10, !range !23, !noalias !58, !noundef !24
  %14 = trunc nuw i8 %13 to i1
  br i1 %14, label %22, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %17 = load i8, ptr %16, align 8, !tbaa !10, !range !23, !noalias !58, !noundef !24
  %18 = zext nneg i8 %17 to i64
  br label %22

19:                                               ; preds = %7, %3
  %20 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !58
  %21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %20, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.19) #16, !noalias !58
  tail call void @abort() #14, !noalias !58
  unreachable

22:                                               ; preds = %11, %15
  %23 = phi i64 [ 1, %11 ], [ %18, %15 ]
  store i64 3, ptr %0, align 8, !tbaa !5
  %24 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %23, ptr %24, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_display(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5
  switch i64 %5, label %23 [
    i64 1, label %6
    i64 2, label %11
    i64 3, label %16
  ]

6:                                                ; preds = %3
  %7 = load ptr, ptr @stdout, align 8, !tbaa !12
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %9 = load i64, ptr %8, align 8, !tbaa !10
  %10 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.20, i64 noundef %9) #15
  br label %26

11:                                               ; preds = %3
  %12 = load ptr, ptr @stdout, align 8, !tbaa !12
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %14 = load double, ptr %13, align 8
  %15 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %12, ptr noundef nonnull @.str.21, double noundef %14) #15
  br label %26

16:                                               ; preds = %3
  %17 = load ptr, ptr @stdout, align 8, !tbaa !12
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %19 = load i8, ptr %18, align 8, !tbaa !10, !range !23, !noundef !24
  %20 = trunc nuw i8 %19 to i1
  %21 = select i1 %20, ptr @.str.22, ptr @.str.23
  %22 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %17, ptr noundef nonnull %21) #15
  br label %26

23:                                               ; preds = %3
  %24 = load ptr, ptr @stderr, align 8, !tbaa !12
  %25 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %24, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.24) #16
  tail call void @abort() #14
  unreachable

26:                                               ; preds = %6, %11, %16
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_newline(ptr nocapture noundef writeonly initializes((0, 8)) %0, ptr nocapture noundef readnone %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = load ptr, ptr @stdout, align 8, !tbaa !12
  %5 = tail call i32 @fputc(i32 10, ptr %4)
  store i64 0, ptr %0, align 8, !tbaa !5
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #7 {
  %1 = alloca %"struct.chirart::Var", align 8
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1) #15
  store i64 0, ptr %1, align 8, !tbaa !5
  call void @chiracg_main(ptr noundef nonnull %1, ptr noundef null, ptr noundef null)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1) #15
  ret i32 0
}

declare void @chiracg_main(ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #9

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #10

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr nocapture noundef, i64 noundef, i64 noundef, ptr nocapture noundef) local_unnamed_addr #11

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, ptr nocapture noundef) local_unnamed_addr #11

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #12

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { alwaysinline mustprogress nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { cold nofree noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { nofree nounwind }
attributes #12 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #13 = { cold }
attributes #14 = { noreturn nounwind }
attributes #15 = { nounwind }
attributes #16 = { cold nounwind }

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
!15 = !{!16, !16, i64 0}
!16 = !{!"p1 _ZTSN7chirart3VarE", !14, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"long", !8, i64 0}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZN7chirart3VarclEPNS_7ArgListE: argument 0"}
!21 = distinct !{!21, !"_ZN7chirart3VarclEPNS_7ArgListE"}
!22 = !{!"branch_weights", !"expected", i32 1, i32 2000}
!23 = !{i8 0, i8 2}
!24 = !{}
!25 = !{!26}
!26 = distinct !{!26, !27, !"_ZN7chirartplERKNS_3VarES2_: argument 0"}
!27 = distinct !{!27, !"_ZN7chirartplERKNS_3VarES2_"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"_ZN7chirartmiERKNS_3VarES2_: argument 0"}
!30 = distinct !{!30, !"_ZN7chirartmiERKNS_3VarES2_"}
!31 = !{!32}
!32 = distinct !{!32, !33, !"_ZN7chirartmlERKNS_3VarES2_: argument 0"}
!33 = distinct !{!33, !"_ZN7chirartmlERKNS_3VarES2_"}
!34 = !{!35}
!35 = distinct !{!35, !36, !"_ZN7chirartdvERKNS_3VarES2_: argument 0"}
!36 = distinct !{!36, !"_ZN7chirartdvERKNS_3VarES2_"}
!37 = !{!38}
!38 = distinct !{!38, !39, !"_ZN7chirartltERKNS_3VarES2_: argument 0"}
!39 = distinct !{!39, !"_ZN7chirartltERKNS_3VarES2_"}
!40 = !{!41}
!41 = distinct !{!41, !42, !"_ZN7chirartleERKNS_3VarES2_: argument 0"}
!42 = distinct !{!42, !"_ZN7chirartleERKNS_3VarES2_"}
!43 = !{!44}
!44 = distinct !{!44, !45, !"_ZN7chirartgtERKNS_3VarES2_: argument 0"}
!45 = distinct !{!45, !"_ZN7chirartgtERKNS_3VarES2_"}
!46 = !{!47}
!47 = distinct !{!47, !48, !"_ZN7chirartgeERKNS_3VarES2_: argument 0"}
!48 = distinct !{!48, !"_ZN7chirartgeERKNS_3VarES2_"}
!49 = !{!50}
!50 = distinct !{!50, !51, !"_ZN7chirarteqERKNS_3VarES2_: argument 0"}
!51 = distinct !{!51, !"_ZN7chirarteqERKNS_3VarES2_"}
!52 = !{!53}
!53 = distinct !{!53, !54, !"_ZN7chirart3VarntEv: argument 0"}
!54 = distinct !{!54, !"_ZN7chirart3VarntEv"}
!55 = !{!56}
!56 = distinct !{!56, !57, !"_ZN7chirartaaERKNS_3VarES2_: argument 0"}
!57 = distinct !{!57, !"_ZN7chirartaaERKNS_3VarES2_"}
!58 = !{!59}
!59 = distinct !{!59, !60, !"_ZN7chirartooERKNS_3VarES2_: argument 0"}
!60 = distinct !{!60, !"_ZN7chirartooERKNS_3VarES2_"}
