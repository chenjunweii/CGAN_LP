// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: annotationdb.proto

#ifndef PROTOBUF_annotationdb_2eproto__INCLUDED
#define PROTOBUF_annotationdb_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 2005000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 2005000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace annotationdb {

// Internal implementation detail -- do not call these.
void  protobuf_AddDesc_annotationdb_2eproto();
void protobuf_AssignDesc_annotationdb_2eproto();
void protobuf_ShutdownFile_annotationdb_2eproto();

class proposal;
class image;

// ===================================================================

class proposal : public ::google::protobuf::Message {
 public:
  proposal();
  virtual ~proposal();

  proposal(const proposal& from);

  inline proposal& operator=(const proposal& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const proposal& default_instance();

  void Swap(proposal* other);

  // implements Message ----------------------------------------------

  proposal* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const proposal& from);
  void MergeFrom(const proposal& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required int32 xmin = 1;
  inline bool has_xmin() const;
  inline void clear_xmin();
  static const int kXminFieldNumber = 1;
  inline ::google::protobuf::int32 xmin() const;
  inline void set_xmin(::google::protobuf::int32 value);

  // required int32 ymin = 2;
  inline bool has_ymin() const;
  inline void clear_ymin();
  static const int kYminFieldNumber = 2;
  inline ::google::protobuf::int32 ymin() const;
  inline void set_ymin(::google::protobuf::int32 value);

  // required int32 w = 3;
  inline bool has_w() const;
  inline void clear_w();
  static const int kWFieldNumber = 3;
  inline ::google::protobuf::int32 w() const;
  inline void set_w(::google::protobuf::int32 value);

  // required int32 h = 4;
  inline bool has_h() const;
  inline void clear_h();
  static const int kHFieldNumber = 4;
  inline ::google::protobuf::int32 h() const;
  inline void set_h(::google::protobuf::int32 value);

  // required string c = 5;
  inline bool has_c() const;
  inline void clear_c();
  static const int kCFieldNumber = 5;
  inline const ::std::string& c() const;
  inline void set_c(const ::std::string& value);
  inline void set_c(const char* value);
  inline void set_c(const char* value, size_t size);
  inline ::std::string* mutable_c();
  inline ::std::string* release_c();
  inline void set_allocated_c(::std::string* c);

  // required string entry = 6;
  inline bool has_entry() const;
  inline void clear_entry();
  static const int kEntryFieldNumber = 6;
  inline const ::std::string& entry() const;
  inline void set_entry(const ::std::string& value);
  inline void set_entry(const char* value);
  inline void set_entry(const char* value, size_t size);
  inline ::std::string* mutable_entry();
  inline ::std::string* release_entry();
  inline void set_allocated_entry(::std::string* entry);

  // @@protoc_insertion_point(class_scope:annotationdb.proposal)
 private:
  inline void set_has_xmin();
  inline void clear_has_xmin();
  inline void set_has_ymin();
  inline void clear_has_ymin();
  inline void set_has_w();
  inline void clear_has_w();
  inline void set_has_h();
  inline void clear_has_h();
  inline void set_has_c();
  inline void clear_has_c();
  inline void set_has_entry();
  inline void clear_has_entry();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::google::protobuf::int32 xmin_;
  ::google::protobuf::int32 ymin_;
  ::google::protobuf::int32 w_;
  ::google::protobuf::int32 h_;
  ::std::string* c_;
  ::std::string* entry_;

  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(6 + 31) / 32];

  friend void  protobuf_AddDesc_annotationdb_2eproto();
  friend void protobuf_AssignDesc_annotationdb_2eproto();
  friend void protobuf_ShutdownFile_annotationdb_2eproto();

  void InitAsDefaultInstance();
  static proposal* default_instance_;
};
// -------------------------------------------------------------------

class image : public ::google::protobuf::Message {
 public:
  image();
  virtual ~image();

  image(const image& from);

  inline image& operator=(const image& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _unknown_fields_;
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return &_unknown_fields_;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const image& default_instance();

  void Swap(image* other);

  // implements Message ----------------------------------------------

  image* New() const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const image& from);
  void MergeFrom(const image& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required string entry = 1;
  inline bool has_entry() const;
  inline void clear_entry();
  static const int kEntryFieldNumber = 1;
  inline const ::std::string& entry() const;
  inline void set_entry(const ::std::string& value);
  inline void set_entry(const char* value);
  inline void set_entry(const char* value, size_t size);
  inline ::std::string* mutable_entry();
  inline ::std::string* release_entry();
  inline void set_allocated_entry(::std::string* entry);

  // repeated string object = 2;
  inline int object_size() const;
  inline void clear_object();
  static const int kObjectFieldNumber = 2;
  inline const ::std::string& object(int index) const;
  inline ::std::string* mutable_object(int index);
  inline void set_object(int index, const ::std::string& value);
  inline void set_object(int index, const char* value);
  inline void set_object(int index, const char* value, size_t size);
  inline ::std::string* add_object();
  inline void add_object(const ::std::string& value);
  inline void add_object(const char* value);
  inline void add_object(const char* value, size_t size);
  inline const ::google::protobuf::RepeatedPtrField< ::std::string>& object() const;
  inline ::google::protobuf::RepeatedPtrField< ::std::string>* mutable_object();

  // @@protoc_insertion_point(class_scope:annotationdb.image)
 private:
  inline void set_has_entry();
  inline void clear_has_entry();

  ::google::protobuf::UnknownFieldSet _unknown_fields_;

  ::std::string* entry_;
  ::google::protobuf::RepeatedPtrField< ::std::string> object_;

  mutable int _cached_size_;
  ::google::protobuf::uint32 _has_bits_[(2 + 31) / 32];

  friend void  protobuf_AddDesc_annotationdb_2eproto();
  friend void protobuf_AssignDesc_annotationdb_2eproto();
  friend void protobuf_ShutdownFile_annotationdb_2eproto();

  void InitAsDefaultInstance();
  static image* default_instance_;
};
// ===================================================================


// ===================================================================

// proposal

// required int32 xmin = 1;
inline bool proposal::has_xmin() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void proposal::set_has_xmin() {
  _has_bits_[0] |= 0x00000001u;
}
inline void proposal::clear_has_xmin() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void proposal::clear_xmin() {
  xmin_ = 0;
  clear_has_xmin();
}
inline ::google::protobuf::int32 proposal::xmin() const {
  return xmin_;
}
inline void proposal::set_xmin(::google::protobuf::int32 value) {
  set_has_xmin();
  xmin_ = value;
}

// required int32 ymin = 2;
inline bool proposal::has_ymin() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void proposal::set_has_ymin() {
  _has_bits_[0] |= 0x00000002u;
}
inline void proposal::clear_has_ymin() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void proposal::clear_ymin() {
  ymin_ = 0;
  clear_has_ymin();
}
inline ::google::protobuf::int32 proposal::ymin() const {
  return ymin_;
}
inline void proposal::set_ymin(::google::protobuf::int32 value) {
  set_has_ymin();
  ymin_ = value;
}

// required int32 w = 3;
inline bool proposal::has_w() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void proposal::set_has_w() {
  _has_bits_[0] |= 0x00000004u;
}
inline void proposal::clear_has_w() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void proposal::clear_w() {
  w_ = 0;
  clear_has_w();
}
inline ::google::protobuf::int32 proposal::w() const {
  return w_;
}
inline void proposal::set_w(::google::protobuf::int32 value) {
  set_has_w();
  w_ = value;
}

// required int32 h = 4;
inline bool proposal::has_h() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void proposal::set_has_h() {
  _has_bits_[0] |= 0x00000008u;
}
inline void proposal::clear_has_h() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void proposal::clear_h() {
  h_ = 0;
  clear_has_h();
}
inline ::google::protobuf::int32 proposal::h() const {
  return h_;
}
inline void proposal::set_h(::google::protobuf::int32 value) {
  set_has_h();
  h_ = value;
}

// required string c = 5;
inline bool proposal::has_c() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void proposal::set_has_c() {
  _has_bits_[0] |= 0x00000010u;
}
inline void proposal::clear_has_c() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void proposal::clear_c() {
  if (c_ != &::google::protobuf::internal::kEmptyString) {
    c_->clear();
  }
  clear_has_c();
}
inline const ::std::string& proposal::c() const {
  return *c_;
}
inline void proposal::set_c(const ::std::string& value) {
  set_has_c();
  if (c_ == &::google::protobuf::internal::kEmptyString) {
    c_ = new ::std::string;
  }
  c_->assign(value);
}
inline void proposal::set_c(const char* value) {
  set_has_c();
  if (c_ == &::google::protobuf::internal::kEmptyString) {
    c_ = new ::std::string;
  }
  c_->assign(value);
}
inline void proposal::set_c(const char* value, size_t size) {
  set_has_c();
  if (c_ == &::google::protobuf::internal::kEmptyString) {
    c_ = new ::std::string;
  }
  c_->assign(reinterpret_cast<const char*>(value), size);
}
inline ::std::string* proposal::mutable_c() {
  set_has_c();
  if (c_ == &::google::protobuf::internal::kEmptyString) {
    c_ = new ::std::string;
  }
  return c_;
}
inline ::std::string* proposal::release_c() {
  clear_has_c();
  if (c_ == &::google::protobuf::internal::kEmptyString) {
    return NULL;
  } else {
    ::std::string* temp = c_;
    c_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
    return temp;
  }
}
inline void proposal::set_allocated_c(::std::string* c) {
  if (c_ != &::google::protobuf::internal::kEmptyString) {
    delete c_;
  }
  if (c) {
    set_has_c();
    c_ = c;
  } else {
    clear_has_c();
    c_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  }
}

// required string entry = 6;
inline bool proposal::has_entry() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void proposal::set_has_entry() {
  _has_bits_[0] |= 0x00000020u;
}
inline void proposal::clear_has_entry() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void proposal::clear_entry() {
  if (entry_ != &::google::protobuf::internal::kEmptyString) {
    entry_->clear();
  }
  clear_has_entry();
}
inline const ::std::string& proposal::entry() const {
  return *entry_;
}
inline void proposal::set_entry(const ::std::string& value) {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  entry_->assign(value);
}
inline void proposal::set_entry(const char* value) {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  entry_->assign(value);
}
inline void proposal::set_entry(const char* value, size_t size) {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  entry_->assign(reinterpret_cast<const char*>(value), size);
}
inline ::std::string* proposal::mutable_entry() {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  return entry_;
}
inline ::std::string* proposal::release_entry() {
  clear_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    return NULL;
  } else {
    ::std::string* temp = entry_;
    entry_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
    return temp;
  }
}
inline void proposal::set_allocated_entry(::std::string* entry) {
  if (entry_ != &::google::protobuf::internal::kEmptyString) {
    delete entry_;
  }
  if (entry) {
    set_has_entry();
    entry_ = entry;
  } else {
    clear_has_entry();
    entry_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  }
}

// -------------------------------------------------------------------

// image

// required string entry = 1;
inline bool image::has_entry() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void image::set_has_entry() {
  _has_bits_[0] |= 0x00000001u;
}
inline void image::clear_has_entry() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void image::clear_entry() {
  if (entry_ != &::google::protobuf::internal::kEmptyString) {
    entry_->clear();
  }
  clear_has_entry();
}
inline const ::std::string& image::entry() const {
  return *entry_;
}
inline void image::set_entry(const ::std::string& value) {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  entry_->assign(value);
}
inline void image::set_entry(const char* value) {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  entry_->assign(value);
}
inline void image::set_entry(const char* value, size_t size) {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  entry_->assign(reinterpret_cast<const char*>(value), size);
}
inline ::std::string* image::mutable_entry() {
  set_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    entry_ = new ::std::string;
  }
  return entry_;
}
inline ::std::string* image::release_entry() {
  clear_has_entry();
  if (entry_ == &::google::protobuf::internal::kEmptyString) {
    return NULL;
  } else {
    ::std::string* temp = entry_;
    entry_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
    return temp;
  }
}
inline void image::set_allocated_entry(::std::string* entry) {
  if (entry_ != &::google::protobuf::internal::kEmptyString) {
    delete entry_;
  }
  if (entry) {
    set_has_entry();
    entry_ = entry;
  } else {
    clear_has_entry();
    entry_ = const_cast< ::std::string*>(&::google::protobuf::internal::kEmptyString);
  }
}

// repeated string object = 2;
inline int image::object_size() const {
  return object_.size();
}
inline void image::clear_object() {
  object_.Clear();
}
inline const ::std::string& image::object(int index) const {
  return object_.Get(index);
}
inline ::std::string* image::mutable_object(int index) {
  return object_.Mutable(index);
}
inline void image::set_object(int index, const ::std::string& value) {
  object_.Mutable(index)->assign(value);
}
inline void image::set_object(int index, const char* value) {
  object_.Mutable(index)->assign(value);
}
inline void image::set_object(int index, const char* value, size_t size) {
  object_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
}
inline ::std::string* image::add_object() {
  return object_.Add();
}
inline void image::add_object(const ::std::string& value) {
  object_.Add()->assign(value);
}
inline void image::add_object(const char* value) {
  object_.Add()->assign(value);
}
inline void image::add_object(const char* value, size_t size) {
  object_.Add()->assign(reinterpret_cast<const char*>(value), size);
}
inline const ::google::protobuf::RepeatedPtrField< ::std::string>&
image::object() const {
  return object_;
}
inline ::google::protobuf::RepeatedPtrField< ::std::string>*
image::mutable_object() {
  return &object_;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace annotationdb

#ifndef SWIG
namespace google {
namespace protobuf {


}  // namespace google
}  // namespace protobuf
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_annotationdb_2eproto__INCLUDED
