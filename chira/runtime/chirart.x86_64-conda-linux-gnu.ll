; ModuleID = '/home/twice/projects/chira/chira/runtime/chirart.cpp'
source_filename = "/home/twice/projects/chira/chira/runtime/chirart.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"

%"struct.chirart::Var" = type { i64, %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { ptr, ptr }

$_ZN7chirart3Var5EqualERKS0_S2_ = comdat any

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
@.str.15 = private unnamed_addr constant [37 x i8] c"Invalid type to perform < comparison\00", align 1
@.str.16 = private unnamed_addr constant [38 x i8] c"Invalid type to perform <= comparison\00", align 1
@.str.17 = private unnamed_addr constant [37 x i8] c"Invalid type to perform > comparison\00", align 1
@.str.18 = private unnamed_addr constant [38 x i8] c"Invalid type to perform >= comparison\00", align 1
@.str.19 = private unnamed_addr constant [47 x i8] c"Invalid type to perform numeric equality check\00", align 1
@.str.20 = private unnamed_addr constant [47 x i8] c"Invalid type to perform shallow equality check\00", align 1
@.str.22 = private unnamed_addr constant [18 x i8] c"Var is not a pair\00", align 1
@.str.23 = private unnamed_addr constant [44 x i8] c"Invalid type to perform deep equality check\00", align 1
@.str.24 = private unnamed_addr constant [41 x i8] c"Invalid type to perform logical negation\00", align 1
@.str.25 = private unnamed_addr constant [36 x i8] c"Invalid type to perform logical AND\00", align 1
@.str.26 = private unnamed_addr constant [35 x i8] c"Invalid type to perform logical OR\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@.str.27 = private unnamed_addr constant [4 x i8] c"%ld\00", align 1
@.str.28 = private unnamed_addr constant [4 x i8] c"%lf\00", align 1
@.str.29 = private unnamed_addr constant [3 x i8] c"#t\00", align 1
@.str.30 = private unnamed_addr constant [3 x i8] c"#f\00", align 1
@.str.31 = private unnamed_addr constant [20 x i8] c"Not implemented yet\00", align 1

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
  %8 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %7) #14
  %9 = load ptr, ptr @stderr, align 8, !tbaa !12
  %10 = tail call i64 @fwrite(ptr nonnull @.str, i64 25, i64 1, ptr %9) #14
  %11 = load ptr, ptr @stderr, align 8, !tbaa !12
  %12 = tail call i32 @fputc(i32 10, ptr %11)
  tail call void @abort() #15
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
  %7 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %6) #14
  %8 = load ptr, ptr @stderr, align 8, !tbaa !12
  %9 = tail call i64 @fwrite(ptr nonnull @.str, i64 25, i64 1, ptr %8) #14
  %10 = load ptr, ptr @stderr, align 8, !tbaa !12
  %11 = tail call i32 @fputc(i32 10, ptr %10)
  tail call void @abort() #15
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

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_bool(ptr nocapture noundef writeonly initializes((0, 16)) %0, i1 noundef zeroext %1) local_unnamed_addr #0 {
  %3 = zext i1 %1 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %3, ptr %4, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_string(ptr nocapture noundef writeonly initializes((0, 24)) %0, ptr noundef %1, i64 noundef %2) local_unnamed_addr #0 {
  store i64 4, ptr %0, align 8, !tbaa !5
  %4 = ptrtoint ptr %1 to i64
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %4, ptr %5, align 8, !tbaa !10
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %2, ptr %6, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_pair(ptr nocapture noundef writeonly initializes((0, 24)) %0, ptr noundef %1, ptr noundef %2) local_unnamed_addr #0 {
  store i64 5, ptr %0, align 8, !tbaa !5
  %4 = ptrtoint ptr %1 to i64
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %4, ptr %5, align 8, !tbaa !10
  %6 = ptrtoint ptr %2 to i64
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %6, ptr %7, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_nil(ptr nocapture noundef writeonly initializes((0, 8)) %0) local_unnamed_addr #0 {
  store i64 6, ptr %0, align 8, !tbaa !5
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
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4) #16
  tail call void @llvm.experimental.noalias.scope.decl(metadata !19)
  %5 = load i64, ptr %1, align 8, !tbaa !5, !noalias !19
  %6 = and i64 %5, -65536
  switch i64 %6, label %7 [
    i64 131072, label %14
    i64 65536, label %14
  ]

7:                                                ; preds = %3
  %8 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %9 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %8) #14, !noalias !19
  %10 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %11 = tail call i64 @fwrite(ptr nonnull @.str.5, i64 41, i64 1, ptr %10) #14
  %12 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %13 = tail call i32 @fputc(i32 10, ptr %12), !noalias !19
  tail call void @abort() #15, !noalias !19
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
  %23 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %22) #14, !noalias !19
  %24 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %25 = load i64, ptr %2, align 8, !tbaa !17, !noalias !19
  %26 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %24, ptr noundef nonnull @.str.3, i64 noundef %15, i64 noundef %25) #17, !noalias !19
  %27 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %28 = tail call i32 @fputc(i32 10, ptr %27), !noalias !19
  tail call void @abort() #15, !noalias !19
  unreachable

29:                                               ; preds = %14
  %30 = icmp ult i64 %18, %15
  br i1 %30, label %31, label %39, !prof !22

31:                                               ; preds = %29
  %32 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %33 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %32) #14, !noalias !19
  %34 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %35 = load i64, ptr %2, align 8, !tbaa !17, !noalias !19
  %36 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %34, ptr noundef nonnull @.str.4, i64 noundef %15, i64 noundef %35) #17, !noalias !19
  %37 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !19
  %38 = tail call i32 @fputc(i32 10, ptr %37), !noalias !19
  tail call void @abort() #15, !noalias !19
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
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4) #16
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local zeroext i1 @chirart_get_bool(ptr nocapture noundef readonly %0) local_unnamed_addr #2 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = icmp eq i64 %2, 3
  br i1 %3, label %11, label %4, !prof !11

4:                                                ; preds = %1
  %5 = load ptr, ptr @stderr, align 8, !tbaa !12
  %6 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %5) #14
  %7 = load ptr, ptr @stderr, align 8, !tbaa !12
  %8 = tail call i64 @fwrite(ptr nonnull @.str.6, i64 20, i64 1, ptr %7) #14
  %9 = load ptr, ptr @stderr, align 8, !tbaa !12
  %10 = tail call i32 @fputc(i32 10, ptr %9)
  tail call void @abort() #15
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
  %43 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %42, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.7) #17, !noalias !25
  tail call void @abort() #15, !noalias !25
  unreachable

44:                                               ; preds = %22, %29
  %45 = phi i64 [ %25, %22 ], [ %40, %29 ]
  %46 = phi i64 [ 1, %22 ], [ 2, %29 ]
  store i64 %46, ptr %0, align 8, !tbaa !5
  store i64 %45, ptr %4, align 8, !tbaa !10
  %47 = getelementptr inbounds nuw i8, ptr %17, i64 24
  %48 = icmp eq ptr %47, %8
  %49 = bitcast i64 %45 to double
  br i1 %48, label %12, label %13, !llvm.loop !28
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_sub(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !30
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !30
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !30
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !30
  %16 = sub nsw i64 %13, %15
  br label %42

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %39

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !30
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %39

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !30
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !30
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fsub double %31, %36
  %38 = bitcast double %37 to i64
  br label %42

39:                                               ; preds = %21, %17
  %40 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !30
  %41 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %40, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.11) #17, !noalias !30
  tail call void @abort() #15, !noalias !30
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
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !33
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !33
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !33
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !33
  %16 = mul nsw i64 %15, %13
  br label %42

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %39

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !33
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %39

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !33
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !33
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fmul double %31, %36
  %38 = bitcast double %37 to i64
  br label %42

39:                                               ; preds = %21, %17
  %40 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !33
  %41 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %40, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.12) #17, !noalias !33
  tail call void @abort() #15, !noalias !33
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
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !36
  %6 = add i64 %5, -1
  %7 = icmp ult i64 %6, 2
  br i1 %7, label %8, label %13

8:                                                ; preds = %3
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %10 = load i64, ptr %9, align 8, !tbaa !5, !noalias !36
  %11 = add i64 %10, -1
  %12 = icmp ult i64 %11, 2
  br i1 %12, label %16, label %13

13:                                               ; preds = %8, %3
  %14 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !36
  %15 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %14, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.14) #17, !noalias !36
  tail call void @abort() #15, !noalias !36
  unreachable

16:                                               ; preds = %8
  %17 = icmp eq i64 %10, 2
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %19 = load double, ptr %18, align 8, !noalias !36
  %20 = bitcast double %19 to i64
  %21 = sitofp i64 %20 to double
  %22 = select i1 %17, double %19, double %21
  %23 = fcmp une double %22, 0.000000e+00
  br i1 %23, label %31, label %24, !prof !11

24:                                               ; preds = %16
  %25 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !36
  %26 = tail call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %25) #14, !noalias !36
  %27 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !36
  %28 = tail call i64 @fwrite(ptr nonnull @.str.13, i64 16, i64 1, ptr %27) #14
  %29 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !36
  %30 = tail call i32 @fputc(i32 10, ptr %29), !noalias !36
  tail call void @abort() #15, !noalias !36
  unreachable

31:                                               ; preds = %16
  %32 = icmp eq i64 %5, 2
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %34 = load double, ptr %33, align 8, !noalias !36
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
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !39
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !39
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !39
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !39
  %16 = icmp slt i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !39
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !39
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !39
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp olt double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !39
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.15) #17, !noalias !39
  tail call void @abort() #15, !noalias !39
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
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !42
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !42
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !42
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !42
  %16 = icmp sle i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !42
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !42
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !42
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp ole double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !42
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.16) #17, !noalias !42
  tail call void @abort() #15, !noalias !42
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
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !45
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !45
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !45
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !45
  %16 = icmp sgt i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !45
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !45
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !45
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp ogt double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !45
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.17) #17, !noalias !45
  tail call void @abort() #15, !noalias !45
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
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !48
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !48
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !48
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !48
  %16 = icmp sge i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !48
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !48
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !48
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp oge double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !48
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.18) #17, !noalias !48
  tail call void @abort() #15, !noalias !48
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
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !51
  %7 = icmp eq i64 %6, 1
  br i1 %7, label %8, label %17

8:                                                ; preds = %3
  %9 = load i64, ptr %5, align 8, !tbaa !5, !noalias !51
  %10 = icmp eq i64 %9, 1
  br i1 %10, label %11, label %21

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !51
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %15 = load i64, ptr %14, align 8, !tbaa !10, !noalias !51
  %16 = icmp eq i64 %13, %15
  br label %41

17:                                               ; preds = %3
  %18 = icmp eq i64 %6, 2
  br i1 %18, label %19, label %38

19:                                               ; preds = %17
  %20 = load i64, ptr %5, align 8, !tbaa !5, !noalias !51
  br label %21

21:                                               ; preds = %8, %19
  %22 = phi i64 [ %20, %19 ], [ %9, %8 ]
  %23 = add i64 %22, -1
  %24 = icmp ult i64 %23, 2
  br i1 %24, label %25, label %38

25:                                               ; preds = %21
  %26 = icmp eq i64 %22, 2
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %28 = load double, ptr %27, align 8, !noalias !51
  %29 = bitcast double %28 to i64
  %30 = sitofp i64 %29 to double
  %31 = select i1 %7, double %30, double %28
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !51
  %34 = bitcast double %33 to i64
  %35 = sitofp i64 %34 to double
  %36 = select i1 %26, double %33, double %35
  %37 = fcmp oeq double %31, %36
  br label %41

38:                                               ; preds = %21, %17
  %39 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !51
  %40 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.19) #17, !noalias !51
  tail call void @abort() #15, !noalias !51
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
define dso_local void @chirart_seq(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = load i64, ptr %4, align 8, !tbaa !5, !noalias !54
  switch i64 %6, label %83 [
    i64 2, label %9
    i64 1, label %7
    i64 3, label %38
    i64 6, label %47
    i64 4, label %50
    i64 5, label %65
    i64 0, label %80
  ]

7:                                                ; preds = %3
  %8 = load i64, ptr %5, align 8, !tbaa !5, !noalias !54
  switch i64 %8, label %83 [
    i64 2, label %11
    i64 1, label %18
  ]

9:                                                ; preds = %3
  %10 = load i64, ptr %5, align 8, !tbaa !5, !noalias !54
  switch i64 %10, label %83 [
    i64 2, label %15
    i64 1, label %24
  ]

11:                                               ; preds = %7
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i64, ptr %12, align 8, !noalias !57
  %14 = sitofp i64 %13 to double
  br label %30

15:                                               ; preds = %9
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %17 = load double, ptr %16, align 8, !noalias !57
  br label %30

18:                                               ; preds = %7
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %20 = load i64, ptr %19, align 8, !tbaa !10, !noalias !57
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %22 = load i64, ptr %21, align 8, !tbaa !10, !noalias !57
  %23 = icmp eq i64 %20, %22
  br label %86

24:                                               ; preds = %9
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %26 = load double, ptr %25, align 8, !noalias !57
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %28 = load i64, ptr %27, align 8, !noalias !57
  %29 = sitofp i64 %28 to double
  br label %34

30:                                               ; preds = %15, %11
  %31 = phi double [ %14, %11 ], [ %17, %15 ]
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %33 = load double, ptr %32, align 8, !noalias !57
  br label %34

34:                                               ; preds = %24, %30
  %35 = phi double [ %31, %30 ], [ %26, %24 ]
  %36 = phi double [ %33, %30 ], [ %29, %24 ]
  %37 = fcmp oeq double %35, %36
  br label %86

38:                                               ; preds = %3
  %39 = load i64, ptr %5, align 8, !tbaa !5, !noalias !54
  %40 = icmp eq i64 %39, 3
  br i1 %40, label %41, label %83

41:                                               ; preds = %38
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %43 = load i8, ptr %42, align 8, !tbaa !10, !range !23, !noalias !54, !noundef !24
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %45 = load i8, ptr %44, align 8, !tbaa !10, !range !23, !noalias !54, !noundef !24
  %46 = icmp eq i8 %43, %45
  br label %86

47:                                               ; preds = %3
  %48 = load i64, ptr %5, align 8, !tbaa !5, !noalias !54
  %49 = icmp eq i64 %48, 6
  br i1 %49, label %86, label %83

50:                                               ; preds = %3
  %51 = load i64, ptr %5, align 8, !tbaa !5, !noalias !54
  %52 = icmp eq i64 %51, 4
  br i1 %52, label %53, label %83

53:                                               ; preds = %50
  %54 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %55 = load i64, ptr %54, align 8, !tbaa !10, !noalias !54
  %56 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %57 = load i64, ptr %56, align 8, !tbaa !10, !noalias !54
  %58 = icmp eq i64 %55, %57
  br i1 %58, label %59, label %86

59:                                               ; preds = %53
  %60 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %61 = load ptr, ptr %60, align 8, !tbaa !10, !noalias !54
  %62 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %63 = load ptr, ptr %62, align 8, !tbaa !10, !noalias !54
  %64 = icmp eq ptr %61, %63
  br label %86

65:                                               ; preds = %3
  %66 = load i64, ptr %5, align 8, !tbaa !5, !noalias !54
  %67 = icmp eq i64 %66, 5
  br i1 %67, label %68, label %83

68:                                               ; preds = %65
  %69 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %70 = load ptr, ptr %69, align 8, !tbaa !10, !noalias !54
  %71 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %72 = load ptr, ptr %71, align 8, !tbaa !10, !noalias !54
  %73 = icmp eq ptr %70, %72
  br i1 %73, label %74, label %86

74:                                               ; preds = %68
  %75 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %76 = load ptr, ptr %75, align 8, !tbaa !10, !noalias !54
  %77 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %78 = load ptr, ptr %77, align 8, !tbaa !10, !noalias !54
  %79 = icmp eq ptr %76, %78
  br label %86

80:                                               ; preds = %3
  %81 = load i64, ptr %5, align 8, !tbaa !5, !noalias !54
  %82 = icmp eq i64 %81, 0
  br i1 %82, label %86, label %83

83:                                               ; preds = %3, %9, %7, %38, %47, %50, %65, %80
  %84 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !54
  %85 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %84, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.20) #17, !noalias !54
  tail call void @abort() #15, !noalias !54
  unreachable

86:                                               ; preds = %68, %74, %53, %59, %80, %47, %18, %34, %41
  %87 = phi i1 [ %46, %41 ], [ %23, %18 ], [ %37, %34 ], [ true, %47 ], [ false, %80 ], [ false, %53 ], [ %64, %59 ], [ false, %68 ], [ %79, %74 ]
  %88 = zext i1 %87 to i64
  store i64 3, ptr %0, align 8, !tbaa !5
  %89 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i64 %88, ptr %89, align 8, !tbaa !10
  ret void
}

; Function Attrs: alwaysinline mustprogress uwtable
define dso_local void @chirart_deq(ptr nocapture noundef writeonly initializes((0, 24)) %0, ptr noundef %1, ptr nocapture noundef readnone %2) local_unnamed_addr #6 {
  %4 = alloca %"struct.chirart::Var", align 16
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 32
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4) #16
  call void @_ZN7chirart3Var5EqualERKS0_S2_(ptr dead_on_unwind nonnull writable sret(%"struct.chirart::Var") align 8 %4, ptr noundef nonnull align 8 dereferenceable(24) %5, ptr noundef nonnull align 8 dereferenceable(24) %6)
  %7 = load <2 x i64>, ptr %4, align 16, !tbaa !10
  store <2 x i64> %7, ptr %0, align 8, !tbaa !10
  %8 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %9 = load i64, ptr %8, align 16, !tbaa !10
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i64 %9, ptr %10, align 8, !tbaa !10
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4) #16
  ret void
}

; Function Attrs: alwaysinline mustprogress uwtable
define linkonce_odr dso_local void @_ZN7chirart3Var5EqualERKS0_S2_(ptr dead_on_unwind noalias writable sret(%"struct.chirart::Var") align 8 %0, ptr noundef nonnull align 8 dereferenceable(24) %1, ptr noundef nonnull align 8 dereferenceable(24) %2) local_unnamed_addr #6 comdat align 2 {
  %4 = alloca %"struct.chirart::Var", align 8
  %5 = alloca %"struct.chirart::Var", align 8
  %6 = load i64, ptr %1, align 8, !tbaa !5
  switch i64 %6, label %131 [
    i64 2, label %9
    i64 1, label %7
    i64 3, label %42
    i64 6, label %53
    i64 4, label %58
    i64 5, label %78
    i64 0, label %126
  ]

7:                                                ; preds = %3
  %8 = load i64, ptr %2, align 8, !tbaa !5
  switch i64 %8, label %131 [
    i64 2, label %11
    i64 1, label %18
  ]

9:                                                ; preds = %3
  %10 = load i64, ptr %2, align 8, !tbaa !5
  switch i64 %10, label %131 [
    i64 2, label %15
    i64 1, label %26
  ]

11:                                               ; preds = %7
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %13 = load i64, ptr %12, align 8, !noalias !60
  %14 = sitofp i64 %13 to double
  br label %32

15:                                               ; preds = %9
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %17 = load double, ptr %16, align 8, !noalias !60
  br label %32

18:                                               ; preds = %7
  tail call void @llvm.experimental.noalias.scope.decl(metadata !60)
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %20 = load i64, ptr %19, align 8, !tbaa !10, !noalias !60
  %21 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %22 = load i64, ptr %21, align 8, !tbaa !10, !noalias !60
  %23 = icmp eq i64 %20, %22
  %24 = zext i1 %23 to i8
  store i64 3, ptr %0, align 8, !tbaa !5, !alias.scope !60
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %24, ptr %25, align 8, !tbaa !10, !alias.scope !60
  br label %134

26:                                               ; preds = %9
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %28 = load double, ptr %27, align 8, !noalias !60
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %30 = load i64, ptr %29, align 8, !noalias !60
  %31 = sitofp i64 %30 to double
  br label %36

32:                                               ; preds = %15, %11
  %33 = phi double [ %14, %11 ], [ %17, %15 ]
  %34 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %35 = load double, ptr %34, align 8, !noalias !60
  br label %36

36:                                               ; preds = %26, %32
  %37 = phi double [ %33, %32 ], [ %28, %26 ]
  %38 = phi double [ %35, %32 ], [ %31, %26 ]
  %39 = fcmp oeq double %37, %38
  %40 = zext i1 %39 to i8
  store i64 3, ptr %0, align 8, !tbaa !5, !alias.scope !60
  %41 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %40, ptr %41, align 8, !tbaa !10, !alias.scope !60
  br label %134

42:                                               ; preds = %3
  %43 = load i64, ptr %2, align 8, !tbaa !5
  %44 = icmp eq i64 %43, 3
  br i1 %44, label %45, label %131

45:                                               ; preds = %42
  %46 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %47 = load i8, ptr %46, align 8, !tbaa !10, !range !23, !noundef !24
  %48 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %49 = load i8, ptr %48, align 8, !tbaa !10, !range !23, !noundef !24
  %50 = icmp eq i8 %47, %49
  %51 = zext i1 %50 to i8
  store i64 3, ptr %0, align 8, !tbaa !5
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %51, ptr %52, align 8, !tbaa !10
  br label %134

53:                                               ; preds = %3
  %54 = load i64, ptr %2, align 8, !tbaa !5
  %55 = icmp eq i64 %54, 6
  br i1 %55, label %56, label %131

56:                                               ; preds = %53
  store i64 3, ptr %0, align 8, !tbaa !5
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 1, ptr %57, align 8, !tbaa !10
  br label %134

58:                                               ; preds = %3
  %59 = load i64, ptr %2, align 8, !tbaa !5
  %60 = icmp eq i64 %59, 4
  br i1 %60, label %61, label %131

61:                                               ; preds = %58
  %62 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %63 = load i64, ptr %62, align 8, !tbaa !10
  %64 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %65 = load i64, ptr %64, align 8, !tbaa !10
  %66 = icmp eq i64 %63, %65
  br i1 %66, label %67, label %75

67:                                               ; preds = %61
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %69 = load ptr, ptr %68, align 8, !tbaa !10
  %70 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %71 = load ptr, ptr %70, align 8, !tbaa !10
  %72 = tail call i32 @bcmp(ptr %69, ptr %71, i64 %63)
  %73 = icmp eq i32 %72, 0
  %74 = zext i1 %73 to i8
  br label %75

75:                                               ; preds = %67, %61
  %76 = phi i8 [ 0, %61 ], [ %74, %67 ]
  store i64 3, ptr %0, align 8, !tbaa !5
  %77 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %76, ptr %77, align 8, !tbaa !10
  br label %134

78:                                               ; preds = %3
  %79 = load i64, ptr %2, align 8, !tbaa !5
  %80 = icmp eq i64 %79, 5
  br i1 %80, label %81, label %131

81:                                               ; preds = %78
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %4) #16
  %82 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %83 = load ptr, ptr %82, align 8, !tbaa !10
  %84 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %85 = load ptr, ptr %84, align 8, !tbaa !10
  call void @_ZN7chirart3Var5EqualERKS0_S2_(ptr dead_on_unwind nonnull writable sret(%"struct.chirart::Var") align 8 %4, ptr noundef nonnull align 8 dereferenceable(24) %83, ptr noundef nonnull align 8 dereferenceable(24) %85)
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %5) #16
  %86 = load i64, ptr %1, align 8, !tbaa !5
  %87 = icmp eq i64 %86, 5
  br i1 %87, label %95, label %88, !prof !11

88:                                               ; preds = %81
  %89 = load ptr, ptr @stderr, align 8, !tbaa !12
  %90 = call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %89) #14
  %91 = load ptr, ptr @stderr, align 8, !tbaa !12
  %92 = call i64 @fwrite(ptr nonnull @.str.22, i64 17, i64 1, ptr %91) #14
  %93 = load ptr, ptr @stderr, align 8, !tbaa !12
  %94 = call i32 @fputc(i32 10, ptr %93)
  call void @abort() #15
  unreachable

95:                                               ; preds = %81
  %96 = load i64, ptr %2, align 8, !tbaa !5
  %97 = icmp eq i64 %96, 5
  br i1 %97, label %105, label %98, !prof !11

98:                                               ; preds = %95
  %99 = load ptr, ptr @stderr, align 8, !tbaa !12
  %100 = call i64 @fwrite(ptr nonnull @.str.1, i64 18, i64 1, ptr %99) #14
  %101 = load ptr, ptr @stderr, align 8, !tbaa !12
  %102 = call i64 @fwrite(ptr nonnull @.str.22, i64 17, i64 1, ptr %101) #14
  %103 = load ptr, ptr @stderr, align 8, !tbaa !12
  %104 = call i32 @fputc(i32 10, ptr %103)
  call void @abort() #15
  unreachable

105:                                              ; preds = %95
  %106 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %107 = load ptr, ptr %106, align 8, !tbaa !10
  %108 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %109 = load ptr, ptr %108, align 8, !tbaa !10
  call void @_ZN7chirart3Var5EqualERKS0_S2_(ptr dead_on_unwind nonnull writable sret(%"struct.chirart::Var") align 8 %5, ptr noundef nonnull align 8 dereferenceable(24) %107, ptr noundef nonnull align 8 dereferenceable(24) %109)
  call void @llvm.experimental.noalias.scope.decl(metadata !63)
  %110 = load i64, ptr %4, align 8, !tbaa !5, !noalias !63
  %111 = icmp eq i64 %110, 3
  %112 = load i64, ptr %5, align 8
  %113 = icmp eq i64 %112, 3
  %114 = select i1 %111, i1 %113, i1 false
  br i1 %114, label %115, label %123

115:                                              ; preds = %105
  %116 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %117 = load i8, ptr %116, align 8, !tbaa !10, !range !23, !noalias !63, !noundef !24
  %118 = trunc nuw i8 %117 to i1
  %119 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %120 = load i8, ptr %119, align 8, !range !23
  %121 = select i1 %118, i8 %120, i8 0
  store i64 3, ptr %0, align 8, !tbaa !5, !alias.scope !63
  %122 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %121, ptr %122, align 8, !tbaa !10, !alias.scope !63
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %5) #16
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %4) #16
  br label %134

123:                                              ; preds = %105
  %124 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !63
  %125 = call i32 (ptr, ptr, ...) @fprintf(ptr noundef %124, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.25) #17, !noalias !63
  call void @abort() #15, !noalias !63
  unreachable

126:                                              ; preds = %3
  %127 = load i64, ptr %2, align 8, !tbaa !5
  %128 = icmp eq i64 %127, 0
  br i1 %128, label %129, label %131

129:                                              ; preds = %126
  store i64 3, ptr %0, align 8, !tbaa !5
  %130 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 0, ptr %130, align 8, !tbaa !10
  br label %134

131:                                              ; preds = %3, %9, %7, %42, %53, %58, %78, %126
  %132 = load ptr, ptr @stderr, align 8, !tbaa !12
  %133 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %132, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.23) #17
  tail call void @abort() #15
  unreachable

134:                                              ; preds = %36, %18, %129, %115, %75, %56, %45
  ret void
}

; Function Attrs: alwaysinline mustprogress nofree nounwind uwtable
define dso_local void @chirart_not(ptr nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readnone %2) local_unnamed_addr #2 {
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !66
  %6 = icmp eq i64 %5, 3
  br i1 %6, label %10, label %7

7:                                                ; preds = %3
  %8 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !66
  %9 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %8, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.24) #17, !noalias !66
  tail call void @abort() #15, !noalias !66
  unreachable

10:                                               ; preds = %3
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %12 = load i8, ptr %11, align 8, !tbaa !10, !range !23, !noalias !66, !noundef !24
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
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !69
  %6 = icmp eq i64 %5, 3
  br i1 %6, label %7, label %19

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %9 = load i64, ptr %8, align 8, !tbaa !5, !noalias !69
  %10 = icmp eq i64 %9, 3
  br i1 %10, label %11, label %19

11:                                               ; preds = %7
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i8, ptr %12, align 8, !tbaa !10, !range !23, !noalias !69, !noundef !24
  %14 = trunc nuw i8 %13 to i1
  br i1 %14, label %15, label %22

15:                                               ; preds = %11
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %17 = load i8, ptr %16, align 8, !tbaa !10, !range !23, !noalias !69, !noundef !24
  %18 = zext nneg i8 %17 to i64
  br label %22

19:                                               ; preds = %7, %3
  %20 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !69
  %21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %20, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.25) #17, !noalias !69
  tail call void @abort() #15, !noalias !69
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
  %5 = load i64, ptr %4, align 8, !tbaa !5, !noalias !72
  %6 = icmp eq i64 %5, 3
  br i1 %6, label %7, label %19

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %9 = load i64, ptr %8, align 8, !tbaa !5, !noalias !72
  %10 = icmp eq i64 %9, 3
  br i1 %10, label %11, label %19

11:                                               ; preds = %7
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load i8, ptr %12, align 8, !tbaa !10, !range !23, !noalias !72, !noundef !24
  %14 = trunc nuw i8 %13 to i1
  br i1 %14, label %22, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %17 = load i8, ptr %16, align 8, !tbaa !10, !range !23, !noalias !72, !noundef !24
  %18 = zext nneg i8 %17 to i64
  br label %22

19:                                               ; preds = %7, %3
  %20 = load ptr, ptr @stderr, align 8, !tbaa !12, !noalias !72
  %21 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %20, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.26) #17, !noalias !72
  tail call void @abort() #15, !noalias !72
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
  %10 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %7, ptr noundef nonnull @.str.27, i64 noundef %9) #16
  br label %26

11:                                               ; preds = %3
  %12 = load ptr, ptr @stdout, align 8, !tbaa !12
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %14 = load double, ptr %13, align 8
  %15 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %12, ptr noundef nonnull @.str.28, double noundef %14) #16
  br label %26

16:                                               ; preds = %3
  %17 = load ptr, ptr @stdout, align 8, !tbaa !12
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %19 = load i8, ptr %18, align 8, !tbaa !10, !range !23, !noundef !24
  %20 = trunc nuw i8 %19 to i1
  %21 = select i1 %20, ptr @.str.29, ptr @.str.30
  %22 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %17, ptr noundef nonnull %21) #16
  br label %26

23:                                               ; preds = %3
  %24 = load ptr, ptr @stderr, align 8, !tbaa !12
  %25 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %24, ptr noundef nonnull @.str.10, ptr noundef nonnull @.str.31) #17
  tail call void @abort() #15
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
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %1) #16
  store i64 0, ptr %1, align 8, !tbaa !5
  call void @chiracg_main(ptr noundef nonnull %1, ptr noundef null, ptr noundef null)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %1) #16
  ret i32 0
}

declare void @chiracg_main(ptr noundef, ptr noundef, ptr noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) local_unnamed_addr #9

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #10

; Function Attrs: nofree nounwind willreturn memory(argmem: read)
declare i32 @bcmp(ptr nocapture, ptr nocapture, i64) local_unnamed_addr #11

; Function Attrs: nofree nounwind
declare noundef i64 @fwrite(ptr nocapture noundef, i64 noundef, i64 noundef, ptr nocapture noundef) local_unnamed_addr #12

; Function Attrs: nofree nounwind
declare noundef i32 @fputc(i32 noundef, ptr nocapture noundef) local_unnamed_addr #12

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #13

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
attributes #11 = { nofree nounwind willreturn memory(argmem: read) }
attributes #12 = { nofree nounwind }
attributes #13 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #14 = { cold }
attributes #15 = { noreturn nounwind }
attributes #16 = { nounwind }
attributes #17 = { cold nounwind }

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
!28 = distinct !{!28, !29}
!29 = !{!"llvm.loop.unroll.enable"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"_ZN7chirartmiERKNS_3VarES2_: argument 0"}
!32 = distinct !{!32, !"_ZN7chirartmiERKNS_3VarES2_"}
!33 = !{!34}
!34 = distinct !{!34, !35, !"_ZN7chirartmlERKNS_3VarES2_: argument 0"}
!35 = distinct !{!35, !"_ZN7chirartmlERKNS_3VarES2_"}
!36 = !{!37}
!37 = distinct !{!37, !38, !"_ZN7chirartdvERKNS_3VarES2_: argument 0"}
!38 = distinct !{!38, !"_ZN7chirartdvERKNS_3VarES2_"}
!39 = !{!40}
!40 = distinct !{!40, !41, !"_ZN7chirartltERKNS_3VarES2_: argument 0"}
!41 = distinct !{!41, !"_ZN7chirartltERKNS_3VarES2_"}
!42 = !{!43}
!43 = distinct !{!43, !44, !"_ZN7chirartleERKNS_3VarES2_: argument 0"}
!44 = distinct !{!44, !"_ZN7chirartleERKNS_3VarES2_"}
!45 = !{!46}
!46 = distinct !{!46, !47, !"_ZN7chirartgtERKNS_3VarES2_: argument 0"}
!47 = distinct !{!47, !"_ZN7chirartgtERKNS_3VarES2_"}
!48 = !{!49}
!49 = distinct !{!49, !50, !"_ZN7chirartgeERKNS_3VarES2_: argument 0"}
!50 = distinct !{!50, !"_ZN7chirartgeERKNS_3VarES2_"}
!51 = !{!52}
!52 = distinct !{!52, !53, !"_ZN7chirarteqERKNS_3VarES2_: argument 0"}
!53 = distinct !{!53, !"_ZN7chirarteqERKNS_3VarES2_"}
!54 = !{!55}
!55 = distinct !{!55, !56, !"_ZN7chirart3Var2EqERKS0_S2_: argument 0"}
!56 = distinct !{!56, !"_ZN7chirart3Var2EqERKS0_S2_"}
!57 = !{!58, !55}
!58 = distinct !{!58, !59, !"_ZN7chirarteqERKNS_3VarES2_: argument 0"}
!59 = distinct !{!59, !"_ZN7chirarteqERKNS_3VarES2_"}
!60 = !{!61}
!61 = distinct !{!61, !62, !"_ZN7chirarteqERKNS_3VarES2_: argument 0"}
!62 = distinct !{!62, !"_ZN7chirarteqERKNS_3VarES2_"}
!63 = !{!64}
!64 = distinct !{!64, !65, !"_ZN7chirartaaERKNS_3VarES2_: argument 0"}
!65 = distinct !{!65, !"_ZN7chirartaaERKNS_3VarES2_"}
!66 = !{!67}
!67 = distinct !{!67, !68, !"_ZN7chirart3VarntEv: argument 0"}
!68 = distinct !{!68, !"_ZN7chirart3VarntEv"}
!69 = !{!70}
!70 = distinct !{!70, !71, !"_ZN7chirartaaERKNS_3VarES2_: argument 0"}
!71 = distinct !{!71, !"_ZN7chirartaaERKNS_3VarES2_"}
!72 = !{!73}
!73 = distinct !{!73, !74, !"_ZN7chirartooERKNS_3VarES2_: argument 0"}
!74 = distinct !{!74, !"_ZN7chirartooERKNS_3VarES2_"}
