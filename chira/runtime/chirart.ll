; ModuleID = '../chira/runtime/chirart.cpp'
source_filename = "../chira/runtime/chirart.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"

@.str = private unnamed_addr constant [27 x i8] c"false && \22Invalid Var tag\22\00", align 1
@.str.1 = private unnamed_addr constant [29 x i8] c"../chira/runtime/chirart.cpp\00", align 1
@__PRETTY_FUNCTION__._ZN7chirart3Var8copyDataERKS0_ = private unnamed_addr constant [32 x i8] c"void Var::copyData(const Var &)\00", align 1
@.str.2 = private unnamed_addr constant [38 x i8] c"isClosure() && \22Var is not a closure\22\00", align 1
@__PRETTY_FUNCTION__._ZNK7chirart3Var10getFuncPtrEv = private unnamed_addr constant [31 x i8] c"Lambda Var::getFuncPtr() const\00", align 1
@__PRETTY_FUNCTION__._ZNK7chirart3Var7getCapsEv = private unnamed_addr constant [25 x i8] c"Env Var::getCaps() const\00", align 1
@.str.3 = private unnamed_addr constant [35 x i8] c"isBool() && \22Var is not a boolean\22\00", align 1
@__PRETTY_FUNCTION__._ZNK7chirart3Var7getBoolEv = private unnamed_addr constant [26 x i8] c"bool Var::getBool() const\00", align 1
@.str.4 = private unnamed_addr constant [31 x i8] c"false && \22Not implemented yet\22\00", align 1
@__PRETTY_FUNCTION__._ZN7chirartplERKNS_3VarES2_ = private unnamed_addr constant [40 x i8] c"Var operator+(const Var &, const Var &)\00", align 1
@__PRETTY_FUNCTION__._ZN7chirartmiERKNS_3VarES2_ = private unnamed_addr constant [40 x i8] c"Var operator-(const Var &, const Var &)\00", align 1
@__PRETTY_FUNCTION__._ZN7chirartltERKNS_3VarES2_ = private unnamed_addr constant [40 x i8] c"Var operator<(const Var &, const Var &)\00", align 1
@.str.6 = private unnamed_addr constant [4 x i8] c"%ld\00", align 1
@__PRETTY_FUNCTION__._ZNK7chirart3Var7DisplayEv = private unnamed_addr constant [26 x i8] c"void Var::Display() const\00", align 1
@.str.7 = private unnamed_addr constant [52 x i8] c"cap_size < (1 << 16) && \22Too many closure captures\22\00", align 1
@__PRETTY_FUNCTION__._ZN7chirart3VarC2EPvPPS0_m = private unnamed_addr constant [30 x i8] c"Var::Var(Lambda, Env, size_t)\00", align 1

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @chirart_unspec() local_unnamed_addr #0 {
  %1 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 0, ptr %1, align 8, !tbaa !5
  ret ptr %1
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @chirart_int(i64 noundef %0) local_unnamed_addr #0 {
  %2 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 1, ptr %2, align 8, !tbaa !5
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i64 %0, ptr %3, align 8, !tbaa !10
  ret ptr %2
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local noalias noundef ptr @chirart_closure(ptr noundef %0, ptr noundef %1, i64 noundef %2) local_unnamed_addr #1 {
  %4 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  %5 = add i64 %2, 65536
  store i64 %5, ptr %4, align 8, !tbaa !5
  %6 = icmp ult i64 %2, 65536
  br i1 %6, label %8, label %7

7:                                                ; preds = %3
  tail call void @__assert_fail(ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.1, i32 noundef 77, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN7chirart3VarC2EPvPPS0_m) #11
  unreachable

8:                                                ; preds = %3
  %9 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %0, ptr %9, align 8, !tbaa !10
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %1, ptr %10, align 8, !tbaa !10
  ret ptr %4
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @chirart_closure_nocap(ptr noundef %0) local_unnamed_addr #0 {
  %2 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 65536, ptr %2, align 8, !tbaa !5
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr %0, ptr %3, align 8, !tbaa !10
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr null, ptr %4, align 8, !tbaa !10
  ret ptr %2
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local void @chirart_set(ptr nocapture noundef writeonly initializes((0, 8)) %0, ptr nocapture noundef readonly %1) local_unnamed_addr #1 {
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
  br label %29

8:                                                ; preds = %2
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %10 = load double, ptr %9, align 8, !tbaa !10
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %10, ptr %11, align 8, !tbaa !10
  br label %29

12:                                               ; preds = %2
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = load i8, ptr %13, align 8, !tbaa !10, !range !11, !noundef !12
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i8 %14, ptr %15, align 8, !tbaa !10
  br label %29

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
  br label %29

26:                                               ; preds = %16
  %27 = icmp eq i64 %3, 0
  br i1 %27, label %29, label %28

28:                                               ; preds = %26
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 66, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN7chirart3Var8copyDataERKS0_) #11
  unreachable

29:                                               ; preds = %4, %8, %12, %19, %26
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @chirart_env(i64 noundef %0) local_unnamed_addr #2 {
  %2 = shl i64 %0, 3
  %3 = tail call noalias noundef ptr @malloc(i64 noundef %2) #10
  ret ptr %3
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local ptr @chirart_get_func_ptr(ptr nocapture noundef readonly %0) local_unnamed_addr #1 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @__assert_fail(ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.1, i32 noundef 113, ptr noundef nonnull @__PRETTY_FUNCTION__._ZNK7chirart3Var10getFuncPtrEv) #11
  unreachable

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !10
  ret ptr %8
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local ptr @chirart_get_caps(ptr nocapture noundef readonly %0) local_unnamed_addr #1 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = and i64 %2, -65536
  %4 = icmp eq i64 %3, 65536
  br i1 %4, label %6, label %5

5:                                                ; preds = %1
  tail call void @__assert_fail(ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.1, i32 noundef 118, ptr noundef nonnull @__PRETTY_FUNCTION__._ZNK7chirart3Var7getCapsEv) #11
  unreachable

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load ptr, ptr %7, align 8, !tbaa !10
  ret ptr %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local ptr @chirart_env_load(ptr nocapture noundef readonly %0, i64 noundef %1) local_unnamed_addr #3 {
  %3 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  %4 = load ptr, ptr %3, align 8, !tbaa !13
  ret ptr %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @chirart_env_store(ptr nocapture noundef writeonly %0, i64 noundef %1, ptr noundef %2) local_unnamed_addr #4 {
  %4 = getelementptr inbounds nuw ptr, ptr %0, i64 %1
  store ptr %2, ptr %4, align 8, !tbaa !13
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local noalias noundef ptr @chirart_copy(ptr nocapture noundef readonly %0) local_unnamed_addr #1 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  switch i64 %2, label %18 [
    i64 1, label %3
    i64 2, label %8
    i64 3, label %13
  ]

3:                                                ; preds = %1
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %5 = load i64, ptr %4, align 8, !tbaa !10
  %6 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 1, ptr %6, align 8, !tbaa !5
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i64 %5, ptr %7, align 8, !tbaa !10
  br label %31

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load double, ptr %9, align 8, !tbaa !10
  %11 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 2, ptr %11, align 8, !tbaa !5
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store double %10, ptr %12, align 8, !tbaa !10
  br label %31

13:                                               ; preds = %1
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %15 = load i8, ptr %14, align 8, !tbaa !10, !range !11, !noundef !12
  %16 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 3, ptr %16, align 8, !tbaa !5
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store i8 %15, ptr %17, align 8, !tbaa !10
  br label %31

18:                                               ; preds = %1
  %19 = and i64 %2, -65536
  %20 = icmp eq i64 %19, 65536
  br i1 %20, label %26, label %21

21:                                               ; preds = %18
  %22 = icmp eq i64 %2, 0
  br i1 %22, label %23, label %25

23:                                               ; preds = %21
  %24 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 0, ptr %24, align 8, !tbaa !5
  br label %31

25:                                               ; preds = %21
  tail call void @__assert_fail(ptr noundef nonnull @.str, ptr noundef nonnull @.str.1, i32 noundef 66, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN7chirart3Var8copyDataERKS0_) #11
  unreachable

26:                                               ; preds = %18
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %28 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 %2, ptr %28, align 8, !tbaa !5
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 8
  %30 = load <2 x ptr>, ptr %27, align 8, !tbaa !10
  store <2 x ptr> %30, ptr %29, align 8, !tbaa !10
  br label %31

31:                                               ; preds = %23, %3, %8, %13, %26
  %32 = phi ptr [ %6, %3 ], [ %11, %8 ], [ %16, %13 ], [ %28, %26 ], [ %24, %23 ]
  ret ptr %32
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local zeroext i1 @chirart_get_bool(ptr nocapture noundef readonly %0) local_unnamed_addr #1 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = icmp eq i64 %2, 3
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @__assert_fail(ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.1, i32 noundef 108, ptr noundef nonnull @__PRETTY_FUNCTION__._ZNK7chirart3Var7getBoolEv) #11
  unreachable

5:                                                ; preds = %1
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load i8, ptr %6, align 8, !tbaa !10, !range !11, !noundef !12
  %8 = trunc nuw i8 %7 to i1
  ret i1 %8
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local noalias noundef ptr @chirart_add(ptr nocapture noundef readonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #1 {
  %3 = load i64, ptr %0, align 8, !tbaa !5, !noalias !16
  %4 = icmp eq i64 %3, 1
  %5 = load i64, ptr %1, align 8, !noalias !16
  %6 = icmp eq i64 %5, 1
  %7 = select i1 %4, i1 %6, i1 false
  br i1 %7, label %9, label %8

8:                                                ; preds = %2
  tail call void @__assert_fail(ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1, i32 noundef 138, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN7chirartplERKNS_3VarES2_) #11, !noalias !16
  unreachable

9:                                                ; preds = %2
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !16
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !16
  %14 = add nsw i64 %13, %11
  %15 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 1, ptr %15, align 8, !tbaa !5
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store i64 %14, ptr %16, align 8, !tbaa !10
  ret ptr %15
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local noalias noundef ptr @chirart_subtract(ptr nocapture noundef readonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #1 {
  %3 = load i64, ptr %0, align 8, !tbaa !5, !noalias !19
  %4 = icmp eq i64 %3, 1
  %5 = load i64, ptr %1, align 8, !noalias !19
  %6 = icmp eq i64 %5, 1
  %7 = select i1 %4, i1 %6, i1 false
  br i1 %7, label %9, label %8

8:                                                ; preds = %2
  tail call void @__assert_fail(ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1, i32 noundef 146, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN7chirartmiERKNS_3VarES2_) #11, !noalias !19
  unreachable

9:                                                ; preds = %2
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !19
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !19
  %14 = sub nsw i64 %11, %13
  %15 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 1, ptr %15, align 8, !tbaa !5
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store i64 %14, ptr %16, align 8, !tbaa !10
  ret ptr %15
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local noalias noundef ptr @chirart_lt(ptr nocapture noundef readonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #1 {
  %3 = load i64, ptr %0, align 8, !tbaa !5, !noalias !22
  %4 = icmp eq i64 %3, 1
  %5 = load i64, ptr %1, align 8, !noalias !22
  %6 = icmp eq i64 %5, 1
  %7 = select i1 %4, i1 %6, i1 false
  br i1 %7, label %9, label %8

8:                                                ; preds = %2
  tail call void @__assert_fail(ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1, i32 noundef 154, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN7chirartltERKNS_3VarES2_) #11, !noalias !22
  unreachable

9:                                                ; preds = %2
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = load i64, ptr %10, align 8, !tbaa !10, !noalias !22
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %13 = load i64, ptr %12, align 8, !tbaa !10, !noalias !22
  %14 = icmp slt i64 %11, %13
  %15 = zext i1 %14 to i8
  %16 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 3, ptr %16, align 8, !tbaa !5
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store i8 %15, ptr %17, align 8, !tbaa !10
  ret ptr %16
}

; Function Attrs: mustprogress nounwind uwtable
define dso_local noalias noundef ptr @chirart_display(ptr nocapture noundef readonly %0) local_unnamed_addr #1 {
  %2 = load i64, ptr %0, align 8, !tbaa !5
  %3 = icmp eq i64 %2, 1
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  tail call void @__assert_fail(ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.1, i32 noundef 163, ptr noundef nonnull @__PRETTY_FUNCTION__._ZNK7chirart3Var7DisplayEv) #11
  unreachable

5:                                                ; preds = %1
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %7 = load i64, ptr %6, align 8, !tbaa !10
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i64 noundef %7)
  %9 = tail call noalias noundef dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #10
  store i64 0, ptr %9, align 8, !tbaa !5
  ret ptr %9
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = tail call ptr @chiracg_main(ptr noundef null)
  ret i32 0
}

declare ptr @chiracg_main(ptr noundef) local_unnamed_addr #6

; Function Attrs: noreturn nounwind
declare void @__assert_fail(ptr noundef, ptr noundef, i32 noundef, ptr noundef) local_unnamed_addr #7

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #9

attributes #0 = { mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { nounwind allocsize(0) }
attributes #11 = { noreturn nounwind }

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
!11 = !{i8 0, i8 2}
!12 = !{}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 _ZTSN7chirart3VarE", !15, i64 0}
!15 = !{!"any pointer", !8, i64 0}
!16 = !{!17}
!17 = distinct !{!17, !18, !"_ZN7chirartplERKNS_3VarES2_: argument 0"}
!18 = distinct !{!18, !"_ZN7chirartplERKNS_3VarES2_"}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZN7chirartmiERKNS_3VarES2_: argument 0"}
!21 = distinct !{!21, !"_ZN7chirartmiERKNS_3VarES2_"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"_ZN7chirartltERKNS_3VarES2_: argument 0"}
!24 = distinct !{!24, !"_ZN7chirartltERKNS_3VarES2_"}
